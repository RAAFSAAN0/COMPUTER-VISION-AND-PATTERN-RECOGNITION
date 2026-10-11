# Real-Time Face Detection and Recognition with Custom CNN Model

This project implements a real-time face detection and recognition system using Haar cascades and a Convolutional Neural Network (CNN). The system performs preprocessing, model training, and real-time face recognition through a webcam interface, all developed in Google Colab.

---

## Features

- **Face Detection**: Utilizes Haar cascades to detect faces in images.
- **Face Preprocessing**: Crops and resizes detected faces to ensure uniform input to the model.
- **CNN Training**: A custom CNN model is trained on the preprocessed dataset for accurate face recognition.
- **Real-Time Recognition**: Recognizes faces in a live video feed via webcam, displaying predictions in real-time.

---

## Project Workflow

### 1. Data Preprocessing
- Faces are detected using Haar cascades.
- Detected faces are cropped, resized to 224x224 pixels, and saved for model training.

### 2. Model Training
- A custom CNN model is designed with layers for feature extraction and classification.
- The model is trained on the preprocessed dataset using TensorFlow's `ImageDataGenerator` for data augmentation.

### 3. Real-Time Recognition
- A webcam video feed is processed using JavaScript for real-time image capture.
- The trained CNN model predicts the class of detected faces.
- Predictions are displayed as an overlay on the video feed.

---

## Technologies Used

- **Google Colab**: For development and model training.
- **OpenCV**: For image processing and face detection.
- **TensorFlow/Keras**: For building and training the CNN model.
- **JavaScript**: For handling live webcam video streams.
- **Python**: For overall implementation.

---

## How to Use

1. Clone or download this repository.
2. Open the `notebook.ipynb` file in Google Colab.
3. Prepare your dataset by organizing images into folders for each class.
4. Update dataset paths and pre-trained model paths in the notebook.
5. Execute the following steps sequentially:
   - Preprocess the data.
   - Train the CNN model.
   - Start the real-time recognition interface.
6. Use your webcam to recognize faces in real-time.

---

## Directory Structure

```
project
├── haarcascade_frontalface_default.xml  # Haar cascade XML for face detection
├── Final_Model.h5                      # Pre-trained CNN model
├── dataset/                            # Original dataset
├── preprocessed/                       # Preprocessed face images
├── notebook.ipynb                      # Main project notebook
```

---

## Requirements

Ensure the following libraries are installed before running the project:

- Python 3.7+
- OpenCV
- TensorFlow
- NumPy
- Matplotlib
- PIL

---

## Model Performance

The model was trained on a dataset containing over 4000 images, achieving robust face recognition performance in real-time.

---

## Contributions

Contributions are welcome! If you encounter any issues or have suggestions for improvement, feel free to create an issue or submit a pull request.

---



