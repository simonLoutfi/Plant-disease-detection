import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from collections import Counter
import cv2
from openai import OpenAI
import os
import numpy as np

api_key = os.getenv('OPENAI_API_KEY')


def fetch_disease_info(disease_name):
    client = OpenAI(api_key=api_key)

    if "healthy" in disease_name.lower():
        context = [
            {"role": "system", "content": "You are a helpful assistant who provides detailed information about plant health."},
            {"role": "user", "content": f"Provide one paragraph of general information about the {disease_name} plant. Start the paragraph with the title 'General Information:'."},
            {"role": "user", "content": f"Provide one paragraph with tips and advice on how to optimize the growth and health of the {disease_name} plant, presented as bullet points. Start the paragraph with the title 'Growth Optimization Tips:'."}
        ]
    else:
        context = [
            {"role": "system", "content": "You are a helpful assistant who provides detailed information about plant diseases."},
            {"role": "user", "content": f"Provide one paragraph explaining what {disease_name} is and detailed information about this disease. Start the paragraph with the title 'Disease Overview:'."},
            {"role": "user", "content": f"Provide one paragraph discussing the reasons why {disease_name} manifests in plants. Start the paragraph with the title 'Causes:'."},
            {"role": "user", "content": f"Provide one paragraph with solutions or treatments to address {disease_name} in plants, presented as bullet points. Start the paragraph with the title 'Treatment and Management:'."}
        ]
    
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=context
    )
    message_content = completion.choices[0].message.content
    return message_content


class CustomImageProcessor:
    def __init__(self, do_normalize=True, do_resize=True, do_random_resized_crop=True, 
                 do_random_horizontal_flip=True, do_center_crop=False, 
                 image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225], 
                 resample=Image.BILINEAR, size=224, crop_size=224, flip_prob=0.5):
        self.do_normalize = do_normalize
        self.do_resize = do_resize
        self.do_random_resized_crop = do_random_resized_crop
        self.do_random_horizontal_flip = do_random_horizontal_flip
        self.do_center_crop = do_center_crop
        self.image_mean = image_mean
        self.image_std = image_std
        self.resample = resample
        self.size = size
        self.crop_size = crop_size
        self.flip_prob = flip_prob
        
        self.transform = self.build_transform()

    def build_transform(self):
        transform_list = []
        
        if self.do_random_resized_crop:
            transform_list.append(transforms.RandomResizedCrop(self.crop_size, interpolation=self.resample))
        
        if self.do_random_horizontal_flip:
            transform_list.append(transforms.RandomHorizontalFlip(p=self.flip_prob))
        
        if self.do_resize:
            transform_list.append(transforms.Resize(self.size, interpolation=self.resample))
        
        if self.do_center_crop:
            transform_list.append(transforms.CenterCrop(self.crop_size))
        
        transform_list.append(transforms.ToTensor())
        
        if self.do_normalize:
            transform_list.append(transforms.Normalize(mean=self.image_mean, std=self.image_std))
        
        return transforms.Compose(transform_list)

    def __call__(self, image):
        return self.transform(image)




# Instantiate the custom image processor
custom_processor = CustomImageProcessor()

# Basic model class definition
class PlantDiseaseClassifier:
    def __init__(self, num_classes):
        self.model = models.efficientnet_b0(pretrained=True)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def classify_image(self, processed_image):
        with torch.no_grad():
            processed_image = processed_image.to(self.device)
            output = self.model(processed_image)
            _, predicted = torch.max(output, 1)
        return predicted.item()


# Instantiate the classifier
classifier = PlantDiseaseClassifier(num_classes=44)

# Load the model weights
checkpoint_path = 'model_epoch_10_acc_0.9255.pth'
classifier.load_model(checkpoint_path)

# Streamlit app header
st.header("Disease Recognition")

def model_prediction(test_image):
    processed_image = custom_processor(test_image).unsqueeze(0)  # Add batch dimension
    predicted_class_index = classifier.classify_image(processed_image)
    return predicted_class_index



# def extract_frames(video_path, interval=5):
#     cap = cv2.VideoCapture(video_path)
#     frame_list = []
#     frame_id = 0
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         if frame_id % interval == 0:
#             frame_list.append(frame)
        
#         frame_id += 1
    
#     cap.release()
#     return frame_list


import tempfile

def extract_frames(video_file, interval=5):
    # Save the video file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(video_file.read())
        temp_video_path = temp_file.name
    
    # Open the video file using the temporary file path
    cap = cv2.VideoCapture(temp_video_path)
    frame_list = []
    frame_id = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_id % interval == 0:
            frame_list.append(frame)
        
        frame_id += 1
    
    cap.release()
    return frame_list


# def model_prediction_video(frames):
#     disease_predictions = []

#     for frame in frames:
#         # Convert the frame to an Image object for processing
#         image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

#         # Process the image using the custom processor
#         processed_image = custom_processor(image)
#         processed_image = processed_image.unsqueeze(0)

#         # Classify the processed image and get the class index
#         predicted_class_index = classifier.classify_image(processed_image)

#         # Map the index to the corresponding class name
#         predicted_class_name = class_names[predicted_class_index]

#         # Append the disease prediction to the list
#         disease_predictions.append(predicted_class_name)
#         st.write(predicted_class_name)

def model_prediction_video(frames):
    disease_predictions = []
    # strawberry_healthy_folder = "Strawberry_healthy_frames"
    # os.makedirs(strawberry_healthy_folder, exist_ok=True)

    for idx, frame in enumerate(frames):
        # Convert the frame to an Image object for processing
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Process the image using the custom processor
        processed_image = custom_processor(image)
        processed_image = processed_image.unsqueeze(0)

        # Classify the processed image and get the class index
        predicted_class_index = classifier.classify_image(processed_image)

        # Map the index to the corresponding class name
        predicted_class_name = class_names[predicted_class_index]

        # Append the disease prediction to the list
        disease_predictions.append(predicted_class_name)
        st.write(predicted_class_name)

        # # Save the frame if the prediction is "Strawberry healthy"
        # if predicted_class_name == "Strawberry healthy":
        #     frame_filename = f"frame_{idx}.jpg"
        #     cv2.imwrite(os.path.join(strawberry_healthy_folder, frame_filename), cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))

    return disease_predictions



    #return disease_predictions


# Define the class names
class_names = [
    'Apple black rot',                       # Index 0
    'Apple healthy',                         # Index 1
    'Apple rust',                            # Index 2
    'Apple scab',                            # Index 3
    'Cassava bacterial blight',              # Index 4
    'Cassava brown streak disease',          # Index 5
    'Cassava healthy',                       # Index 6
    'Cassava mosaic disease',                # Index 7
    'Cherry healthy',                        # Index 8
    'Cherry powdery mildew',                 # Index 9
    'Corn common rust',                      # Index 10
    'Corn gray leaf spot',                   # Index 11
    'Corn healthy',                          # Index 12
    'Corn northern leaf blight',             # Index 13
    'Grape black measles',                   # Index 14
    'Grape black rot',                       # Index 15
    'Grape healthy',                         # Index 16
    'Grape leaf blight (isariopsis leaf spot)', # Index 17
    'Olive aculus olearius',                 # Index 18
    'Olive healthy',                         # Index 19
    'Olive peacock spot',                    # Index 20
    'Peach bacterial spot',                  # Index 21
    'Peach healthy',                         # Index 22
    'Pepper bell bacterial spot',            # Index 23
    'Pepper bell healthy',                   # Index 24
    'Potato early blight',                   # Index 25
    'Potato healthy',                        # Index 26
    'Potato late blight',                    # Index 27
    'Soybean caterpillar',                   # Index 28
    'Soybean diabrotica speciosa',           # Index 29
    'Soybean healthy',                       # Index 30                
    'Strawberry leaf scorch',# Index 31
    'Strawberry healthy',                # Index 32
    'Tea algal leaf',                        # Index 33
    'Tea brown blight',                      # Index 34
    'Tea healthy',                           # Index 35
    'Tea red leaf spot',                     # Index 36
    'Tomato bacterial spot',                 # Index 37
    'Tomato early blight',                   # Index 38
    'Tomato healthy',                        # Index 39
    'Tomato leaf mold',                      # Index 40
    'Wheat brown rust',                      # Index 41
    'Wheat healthy',                         # Index 42
    'Wheat septoria',                        # Index 43
]

# Radio button to select upload type
upload_type = st.radio("Choose Upload Type", ("Single Image", "Multiple Images", "Video"))

if upload_type == "Single Image":
    test_image = st.file_uploader("Choose an Image:", type=['jpg', 'png', 'jpeg'], key="single_image_uploader")

    if test_image and st.button("Show Image"):
        st.image(test_image, use_column_width=True)

    if st.button("Predict"):
        if test_image:
            result_index = model_prediction(Image.open(test_image))
            predicted_disease = class_names[result_index]
            st.success(f"Model is predicting: {predicted_disease}")
            disease_info = fetch_disease_info(predicted_disease)
            st.write(disease_info)
        else:
            st.error("Please upload an image.")

elif upload_type == "Multiple Images":
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []

    uploaded_files = st.file_uploader("Choose Images", accept_multiple_files=True, type=['jpg', 'png', 'jpeg'], key="multiple_images_uploader")

    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files

    if st.button("Show Images"):
        if st.session_state.uploaded_files:
            for uploaded_file in st.session_state.uploaded_files:
                st.image(uploaded_file, use_column_width=True)

    if st.button("Predict"):
        if st.session_state.uploaded_files:
            predictions = []
            for uploaded_file in st.session_state.uploaded_files:
                result_index = model_prediction(Image.open(uploaded_file))
                predicted_disease = class_names[result_index]
                predictions.append(predicted_disease)
            
            total_images = len(predictions)
            class_counts = Counter(predictions)
            class_percentages = {class_name: (count / total_images) * 100 for class_name, count in class_counts.items()}
            
            st.write("Class Distribution in the Uploaded Images:")
            for class_name, percentage in class_percentages.items():
                st.write(f"{class_name}: {percentage:.2f}%")
        else:
            st.error("Please upload at least one image.")
        
elif upload_type == "Video":
    video_file = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"], key="video_uploader")

    if st.button("Process Video"):
        if video_file:
            st.write("Processing video...")
            
            # 1. Extract frames from video
            frames = extract_frames(video_file, interval=30)
            
            # 2. Detect disease in each frame
            disease_predictions = model_prediction_video(frames)
            
            # 3. Output the detected diseases
            counter = Counter(disease_predictions)

            # Find the most common disease
            most_common_disease, count = counter.most_common(1)[0]

            st.write(f"The most common disease detected is '{most_common_disease}' which appears {count} times.")
            disease_info = fetch_disease_info(most_common_disease)
            st.write(disease_info)
            
        else:
            st.error("Please upload a video.")
