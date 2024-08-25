Plant Disease Classification System

Overview
This project is a deep learning-based image classification system designed to detect and classify plant diseases. The system uses a Convolutional Neural Network (CNN) architecture, specifically an EfficientNet-B0 model, fine-tuned to classify various plant diseases from a dataset of plant images. The system can be used to help farmers and agricultural professionals diagnose plant diseases early, potentially improving crop yields and reducing losses.

Dataset
The dataset used in this project is sourced from Kaggle and is specifically tailored for plant disease classification tasks.
Dataset Link: [Plant Disease Classification Merged Dataset](https://www.kaggle.com/datasets/alinedobrovsky/plant-disease-classification-merged-dataset/data)
Classes: The dataset comprises multiple classes, each representing a different plant disease.
Training Set: 1,000 instances per class.
Validation Set: 500 instances per class.
Testing Set: 5% of the original dataset reserved for testing

System Architecture
1. Data Preprocessing:
    Custom Image Processor: A custom image processor is implemented to handle image transformations such as normalization, resizing, random cropping, and horizontal flipping.
    Transformations:
        Random Resized Crop
        Random Horizontal Flip
        Resize
        Center Crop (optional)
        Normalization
2. Model Development:
    Model Architecture: The system utilizes the EfficientNet-B0 architecture for image classification. The model is fine-tuned to classify plant diseases by modifying the classifier layer to output the required number of classes.
    Fine-Tuning:
        All layers of the model are unfrozen for training.
        Different learning rates are applied to different layers of the model (e.g., earlier layers vs. the classifier).
    Loss Function: CrossEntropyLoss is used to calculate the loss during training.
    Optimizer: AdamW optimizer is used for training with support for gradient accumulation.
    Learning Rate Scheduler: A linear learning rate scheduler with a warmup phase is used to adjust the learning rate during training.
    Early Stopping: Implemented to prevent overfitting by monitoring validation accuracy.
3. Training and Evaluation:
    Training: The model is trained on a dataset of labeled plant images. During training, the loss is calculated, and gradients are accumulated before updating the model parameters.
    Validation: After each epoch, the model is evaluated on a validation set. Key metrics such as accuracy, precision, recall, and F1 score are calculated.
    Early Stopping: The training stops early if the validation accuracy does not improve beyond a set threshold, helping to prevent overfitting.
4. Inference:
    Image Classification: The trained model can be used to classify new images of plants. The system returns the predicted disease label for the input image.


User Interface
    The system features a user-friendly interface that allows users to easily upload images or videos for disease recognition.
1. Choosing Upload Type:
    Single Image: Upload a single image for disease recognition.
    Multiple Images: Upload multiple images at once for batch processing.
    Video: Upload a video file for disease recognition.
2. Uploading Files:
    Drag and Drop: Users can drag and drop image or video files into the designated area.
    Browse Files: Alternatively, users can click the "Browse files" button to select files manually.
    Supported Formats: The system supports JPG, PNG, and JPEG formats, with a maximum file size limit of 200MB per file.
3. Processing:
    Show Images: Before making predictions, users can preview the uploaded images to confirm their selection.
    Predict: Clicking the "Predict" button initiates the disease recognition process. The system analyzes the images or video and provides the predicted disease labels.

Usage Instructions
1. Access the Interface:
    Open the disease recognition interface in your web browser or application where it is hosted.
2. Upload Files:
    Select Upload Type: Choose whether to upload a single image, multiple images, or a video.
    Add Files: Drag and drop your files into the upload area or use the "Browse files" button to select them manually.
    Ensure Compatibility: Verify that your files are in the supported formats (JPG, PNG, JPEG) and do not exceed the 200MB size limit.
3. Preview Uploaded Files:
    Click the "Show Images" button to preview the uploaded images. This step ensures that the correct files have been selected for analysis.
4. Run Prediction:
    Once satisfied with the uploaded files, click the "Predict" button to start the disease recognition process.
    The system will process the images or video and display the predicted disease labels along with any relevant confidence scores or additional information.
5. Interpreting Results:
    Review the predicted labels to understand the potential diseases affecting the plants in the uploaded images or video.
    Use this information to make informed decisions regarding plant care and disease management.

Conclusion
This system offers a powerful and user-friendly tool for plant disease classification. By leveraging a fine-tuned EfficientNet-B0 model and an intuitive interface, it enables quick and accurate disease recognition to support agricultural decision-making, ultimately contributing to improved crop management and yields.
