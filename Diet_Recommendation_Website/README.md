# 🥗 Smart Diet Recommendation System using BMI

## 📌 Project Overview
The Smart Diet Recommendation System using BMI is a web-based intelligent application that provides personalized diet plans based on a user’s physical attributes and lifestyle. The system collects user inputs such as age, gender, height, weight, number of meals per day, exercise intensity, and weight goals to generate customized dietary recommendations.

The application automatically calculates the user's **Body Mass Index (BMI)** and predicts the user's health status using a **Random Forest Classifier**. Based on the predicted health condition and fitness goal, the system recommends a balanced diet plan including breakfast, lunch, and dinner. The system also visualizes the distribution of **carbohydrates, proteins, and fats** using a pie chart to help users understand their nutritional intake. :contentReference[oaicite:0]{index=0}

---

## ❗ Problem Statement
Maintaining a balanced and healthy diet has become challenging in today's fast-paced lifestyle. Many individuals lack proper knowledge about their nutritional requirements and often follow generic diet plans that may not suit their body type or health condition.

This project addresses this problem by providing a **personalized diet recommendation system** that analyzes user health parameters and suggests suitable meal plans.

---

## 💡 Proposed Solution
The Smart Diet Recommendation System uses machine learning and health analytics to generate personalized meal recommendations. The system calculates BMI using the user's height and weight and determines the user's health status according to standard health guidelines.

Using this information, the system suggests suitable diet plans aligned with the user's goal such as **weight loss, weight maintenance, or weight gain**. It also analyzes the user's intake of carbohydrates, proteins, and fats and visualizes them through graphical charts.

---

## ⚙️ Technologies Used
- 🐍 **Python** – Backend logic and machine learning implementation  
- 🌐 **Flask** – Web framework for building the application  
- 🧾 **HTML** – Structure of the web interface  
- 🎨 **CSS** – Styling and design of the web pages  
- 📊 **Matplotlib / Seaborn** – Visualization of nutritional data  
- 📂 **Machine Learning Algorithms**
  - Random Forest Classifier (Health status prediction)
  - K-Nearest Neighbors (KNN) (Alternative meal recommendations)

---

## 🧠 Methodology
The system works through the following steps:

1. The user enters personal details such as age, gender, height, weight, exercise level, and meal frequency.
2. The system calculates the **Body Mass Index (BMI)**.
3. A **Random Forest Classifier** predicts the user's health status.
4. Based on the health condition and fitness goals, the system generates personalized meal plans.
5. The application suggests food options for **breakfast, lunch, and dinner**.
6. Macronutrient distribution (carbohydrates, proteins, fats) is displayed using a **pie chart visualization**.

---

## 📊 BMI Formula
BMI is calculated using the following formula:

BMI = Weight (kg) / Height (m²)

Based on BMI values, users are classified into categories such as:
- Underweight
- Normal
- Overweight
- Obese

---

## ✨ Features
✔ Automatic BMI calculation  
✔ Health status prediction using Machine Learning  
✔ Personalized diet plan generation  
✔ Meal suggestions for breakfast, lunch, and dinner  
✔ Macronutrient visualization using pie charts  
✔ User-friendly web interface  

---
