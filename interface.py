import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import cv2
from openai import OpenAI
import os
import tempfile
from collections import Counter

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
    def __init__(self, image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225], size=224):
        self.image_mean = image_mean
        self.image_std = image_std
        self.size = size

    def process_image(self, image):
        image = image.resize((self.size, self.size))
        image = np.array(image) / 255.0
        image = (image - self.image_mean) / self.image_std
        image = np.expand_dims(image, axis=0)
        return image

# Instantiate the custom image processor
custom_processor = CustomImageProcessor()

# Basic model class definition
class PlantDiseaseClassifier:
    def __init__(self, num_classes):
        base_model = tf.keras.applications.EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        self.model = tf.keras.Model(inputs=base_model.input, outputs=output)

    def load_model(self, model_path):
        self.model.load_weights(model_path)

    def classify_image(self, processed_image):
        predictions = self.model.predict(processed_image)
        return np.argmax(predictions, axis=1)[0]

# Instantiate the classifier
classifier = PlantDiseaseClassifier(num_classes=44)

# Load the model weights
checkpoint_path = 'model_epoch_10_acc_0.9255.h5'
classifier.load_model(checkpoint_path)

# Streamlit app header
st.header("Disease Recognition")

def model_prediction(test_image):
    processed_image = custom_processor.process_image(test_image)
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

    for idx, frame in enumerate(frames):
        # Convert the frame to an Image object for processing
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Process the image using the custom processor
        processed_image = custom_processor.process_image(image)

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
    'Strawberry leaf scorch',                # Index 31
    'Strawberry healthy',                    # Index 32
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
                st.write(f"Prediction: {predicted_disease}")

            most_common_disease = Counter(predictions).most_common(1)[0][0]
            disease_info = fetch_disease_info(most_common_disease)
            st.write(disease_info)
        else:
            st.error("Please upload images.")

elif upload_type == "Video":
    video_file = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi"], key="video_uploader")

    if video_file and st.button("Show Video"):
        st.video(video_file, format="video/mp4")

    if video_file and st.button("Process Video"):
        frames = extract_frames(video_file)
        predictions = model_prediction_video(frames)

