# 👁️ Diabetic Retinopathy Detection using Deep Learning

## 📌 Project Overview
This project presents an AI-based Diabetic Retinopathy Detection System developed using Deep Learning and Flask. The system analyzes retinal fundus images and classifies diabetic retinopathy into five stages:

- No_DR
- Mild
- Moderate
- Severe
- Proliferate_DR

The project uses the InceptionV3 Convolutional Neural Network (CNN) model for automated retinal image classification and provides predictions through a Flask-based web application.

---

# 🚀 Features

- Automated diabetic retinopathy detection
- Multi-class retinal disease classification
- Deep learning-based image analysis
- Flask web application interface
- Retinal image upload and prediction
- Real-time classification output
- Performance evaluation using confusion matrix and classification metrics

---

# 🧠 Technologies Used

- Python
- Flask
- PyTorch
- Torchvision
- OpenCV
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

# 📂 Dataset

Download the dataset from Google Drive:

🔗 Dataset Link:  
https://drive.google.com/drive/folders/1-B_J5rCD5d-sCZ1PPShdCc5VwYr3WLYB?usp=sharing

### Dataset Classes
- No_DR
- Mild
- Moderate
- Severe
- Proliferate_DR

### Total Images
- 7143 Retinal Fundus Images

---

# 🧠 Trained Model

Download the trained model file from Google Drive:

🔗 Model Link:  
https://drive.google.com/file/d/1jdSRkmfKeii26MXaY4I2j6IKiovyGqQO/view?usp=drive_link

### Important
After downloading:

Place the model file inside:

```plaintext
model/best_dr_model.pth
```

---

# 📁 Project Structure

```plaintext
DR_Flask_project/
│
├── app.py
├── train_model.py
├── evaluate.py
├── requirements.txt
│
├── model/
│   └── best_dr_model.pth
│
├── uploads/
├── templates/
├── static/
├── augmented_colored/
│
└── README.md
```

---

# ⚙️ Installation

## Step 1: Clone Repository

```bash
git clone <repository_link>
cd DR_Flask_project
```

---

## Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 3: Download Dataset and Model

- Download dataset from the provided Google Drive link
- Download trained model file
- Place dataset inside project folder
- Place model inside `model/` folder

---

# ▶️ Run the Application

```bash
python app.py
```

Open browser:

```plaintext
http://127.0.0.1:5000/
```

---

# 📊 Performance Metrics

The model performance is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

These metrics help evaluate classification performance across all diabetic retinopathy stages.

---

# 🔬 Methodology

1. Retinal Fundus Image Input
2. Image Preprocessing
3. Dataset Preparation
4. Train-Test Split
5. Deep Learning Model Training (InceptionV3)
6. DR Detection and Classification
7. Performance Evaluation
8. Flask Web Application Deployment

---

# 🎯 Project Outcome

The developed system successfully classifies diabetic retinopathy stages from retinal images using deep learning techniques and demonstrates the potential of AI-assisted medical image analysis for ophthalmology applications.

---

# 🚀 Future Scope

- Integration with real-time hospital systems
- Mobile-based DR screening applications
- Explainable AI using Grad-CAM
- Improved balancing using advanced augmentation
- Multi-eye disease detection system
- Deployment on cloud platforms

---

# 👨‍💻 Developed By

Project developed for academic and research purposes in the field of Artificial Intelligence and Medical Image Analysis.
