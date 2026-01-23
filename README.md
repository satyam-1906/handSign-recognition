# ✋ Hand Sign Recognition using Machine Learning & Computer Vision

A machine learning and computer vision project that recognizes **American Sign Language (ASL)** hand signs and predicts the corresponding **alphabet letters in real time** using a webcam.

This project was built as a learning exercise to understand the **complete ML pipeline**, from data preprocessing to live inference.

---

## 📌 Features

- ASL alphabet classification  
- Image-to-tabular data conversion  
- Machine learning–based gesture recognition  
- Real-time hand detection and prediction  
- Webcam-based live inference  

---

## 📂 Dataset

- Source: **Kaggle**
- Contains labeled images of **American Sign Language (ASL)** hand gestures  
- Each image corresponds to a single alphabet letter  

---

## 🔧 Data Preprocessing

The image dataset is converted into a structured CSV-based dataset:

1. Images are loaded using OpenCV  
2. Converted to grayscale to reduce complexity  
3. Resized to a fixed resolution  
4. Pixel values are flattened into a 1D feature vector  
5. Labels are appended to create a tabular dataset  

This allows traditional machine learning models to be trained on image data.

---

## 🤖 Model Used

- **Random Forest Classifier**

### Why Random Forest?
- Handles high-dimensional feature spaces  
- Robust to noise and overfitting  
- Works well with pixel-based image features  

### Training Workflow
- Dataset split into training and testing sets  
- Model trained on pixel-value features  
- Performance evaluated using classification accuracy  

---

## 👁️ Real-Time Prediction Pipeline

1. Webcam feed captured using OpenCV  
2. Hand landmarks detected using **MediaPipe Hands**  
3. Hand region extracted as Region of Interest (ROI)  
4. ROI processed using the same preprocessing steps  
5. Trained model predicts the signed letter  
6. Prediction displayed on live video feed  

---

## 🧠 Core Concepts Covered

- Image preprocessing  
- Feature extraction  
- Supervised learning  
- Ensemble methods (Random Forest)  
- Hand landmark detection  
- Real-time computer vision  

---

## 🛠️ Tech Stack

- Python  
- OpenCV  
- MediaPipe  
- NumPy  
- Pandas  
- Scikit-learn  

---

## 📈 Learning Outcomes

- Learned how image data can be transformed into structured datasets  
- Built and evaluated a machine learning classification model  
- Integrated ML models with real-time video streams  
- Gained hands-on experience with computer vision pipelines  

---

## 🚀 Future Improvements

- Replace pixel-based features with CNN-based deep learning  
- Add support for dynamic gestures and words  
- Apply data augmentation for better accuracy  
- Optimize for real-time performance  

---

## 🙌 Acknowledgements

- Kaggle for the dataset  
- Google MediaPipe for hand tracking  
- Open-source ML and CV libraries  

---

⭐ If you found this project interesting, feel free to star the repository!
