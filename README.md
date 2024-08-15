# Find-Your-Photos

**Find-Your-Photos** is a face recognition application built using PyTorch and OpenCV. It allows you to find photos containing your face within a specified directory by leveraging state-of-the-art deep learning techniques.

## Installation

Before running the application, you need to install the required dependencies:

```bash
pip install facenet-pytorch opencv-python imutils numpy
```

## How It Works

This application utilizes pre-trained models—**MTCNN** and **Inception Resnet V1**—to detect and recognize faces within images.

### 1. Face Detection with MTCNN

- **MTCNN (Multi-Task Cascaded Convolutional Networks)** is used for detecting and extracting face regions of interest (ROIs) from images.
- MTCNN employs traditional bounding box regression to detect and localize faces accurately.
- It also performs tasks like face alignment, ensuring that the detected faces are properly aligned before further processing.

### 2. Face Recognition with Inception Resnet V1

- **Inception Resnet V1** is a pre-trained deep learning model that extracts embeddings from face images detected by MTCNN.
- These embeddings are high-dimensional vectors that uniquely represent facial features, allowing for effective face comparison.
- The model was trained using a Siamese network architecture, where it learned to minimize the distance between embeddings of the same person while maximizing the distance between embeddings of different people.
- A **triplet loss function** was employed during training to ensure the model’s accuracy in distinguishing between different faces.

### 3. Face Matching

- The application compares the embeddings of the captured face from your webcam with those from the images in the provided directory.
- It uses Euclidean distance to measure the similarity between embeddings. Images with distances below a certain threshold are considered as matches.

## Usage

1. **Capture Face**: The application captures an image of your face using your webcam.
2. **Search Directory**: It then searches through the provided directory to find images that contain your face.
3. **Display Results**: The matched images are displayed, allowing you to see where your face appears in the directory.

---
