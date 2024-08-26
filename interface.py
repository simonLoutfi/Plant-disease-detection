import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from collections import Counter
import cv2
import tempfile

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

def model_prediction_video(frames):
    disease_predictions = []

    for frame in frames:
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

    return disease_predictions

# Define the class names
class_names = [
    'Apple black rot', 'Apple healthy', 'Apple rust', 'Apple scab',
    'Cassava bacterial blight', 'Cassava brown streak disease', 'Cassava healthy',
    'Cassava mosaic disease', 'Cherry healthy', 'Cherry powdery mildew',
    'Corn common rust', 'Corn gray leaf spot', 'Corn healthy',
    'Corn northern leaf blight', 'Grape black measles', 'Grape black rot',
    'Grape healthy', 'Grape leaf blight (isariopsis leaf spot)', 'Peach bacterial spot', 'Peach healthy',
    'Pepper bell bacterial spot', 'Pepper bell healthy', 'Potato early blight',
    'Potato healthy', 'Potato late blight', 'Soybean caterpillar',
    'Soybean diabrotica speciosa', 'Soybean healthy', 'Strawberry leaf scorch',
    'Strawberry healthy', 'Tea algal leaf', 'Tea brown blight', 'Tea healthy',
    'Tea red leaf spot', 'Tomato bacterial spot', 'Tomato early blight',
    'Tomato healthy', 'Tomato leaf mold', 'Wheat brown rust', 'Wheat healthy',
    'Wheat septoria'
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
            
        else:
            st.error("Please upload a video.")
