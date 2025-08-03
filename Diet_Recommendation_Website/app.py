from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import random
import os
from threading import Thread
import webbrowser
import time
from werkzeug.serving import make_server
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import io
import base64
app = Flask(__name__)
diet_df = pd.read_csv("final_diet_meal_data.csv")
data = pd.DataFrame({
    'Age': np.random.randint(18, 60, 200),
    'Gender': np.random.choice(['Male', 'Female'], 200),
    'BMI': np.random.uniform(18, 35, 200),
    'Meals_Per_Day': np.random.randint(2, 6, 200),
    'Exercise_Intensity': np.random.choice(['Low', 'Medium', 'High'], 200),
    'Weight_Goal': np.random.choice(['Lose', 'Maintain', 'Gain'], 200),
    'Health_Status': np.random.choice(['Healthy', 'Overweight'], 200)
})
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})
data['Exercise_Intensity'] = data['Exercise_Intensity'].map({'Low': 0, 'Medium': 1, 'High': 2})
data['Weight_Goal'] = data['Weight_Goal'].map({'Lose': 0, 'Maintain': 1, 'Gain': 2})
data['Health_Status'] = data['Health_Status'].map({'Healthy': 0, 'Overweight': 1})

X = data.drop('Health_Status', axis=1)
y = data['Health_Status']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
def classify_bmi(bmi):
    if bmi < 16:
        return "Severely Underweight"
    elif 16 <= bmi < 17:
        return "Moderately Underweight"
    elif 17 <= bmi < 18.5:
        return "Mildly Underweight"
    elif 18.5 <= bmi < 25:
        return "Healthy"
    elif 25 <= bmi < 30:
        return "Overweight"
    elif 30 <= bmi < 35:
        return "Obese Class I"
    elif 35 <= bmi < 40:
        return "Obese Class II"
    else:
        return "Obese Class III"
def predict_health(age, gender, height, weight, meals, exercise, goal):
    bmi = weight / (height / 100) ** 2
    health_status = classify_bmi(bmi)
    return bmi, health_status
def get_meal_plan(goal):
    goal = goal.capitalize()
    filtered = diet_df[diet_df['Goal'] == goal]
    if filtered.empty:
        return {'Breakfast': ("No Data", "N/A"), 'Lunch': ("No Data", "N/A"), 'Dinner': ("No Data", "N/A")}, 0, 0, 0
    try:
        breakfast = filtered[filtered['Meal_Type'] == 'Breakfast'].sample(1).iloc[0]
    except:
        breakfast = {'Food_Item': 'No Data', 'Alternative': 'N/A', 'Carbs': 0, 'Proteins': 0, 'Fats': 0}
    try:
        lunch = filtered[filtered['Meal_Type'] == 'Lunch'].sample(1).iloc[0]
    except:
        lunch = {'Food_Item': 'No Data', 'Alternative': 'N/A', 'Carbs': 0, 'Proteins': 0, 'Fats': 0}
    try:
        dinner = filtered[filtered['Meal_Type'] == 'Dinner'].sample(1).iloc[0]
    except:
        dinner = {'Food_Item': 'No Data', 'Alternative': 'N/A', 'Carbs': 0, 'Proteins': 0, 'Fats': 0}
    total_carbs = breakfast['Carbs'] + lunch['Carbs'] + dinner['Carbs']
    total_proteins = breakfast['Proteins'] + lunch['Proteins'] + dinner['Proteins']
    total_fats = breakfast['Fats'] + lunch['Fats'] + dinner['Fats']
    return {
        'Breakfast': (breakfast['Food_Item'], breakfast['Alternative']),
        'Lunch': (lunch['Food_Item'], lunch['Alternative']),
        'Dinner': (dinner['Food_Item'], dinner['Alternative'])
    }, total_carbs, total_proteins, total_fats


def generate_pie_chart(carbs, proteins, fats):
    fig, ax = plt.subplots()
    ax.pie([carbs, proteins, fats], labels=['Carbs', 'Proteins', 'Fats'], autopct='%1.1f%%')
    ax.set_title('Daily Nutritional Distribution (Carbs, Proteins, Fats)')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    try:
        age = int(request.form['age'])
        gender = request.form['gender']
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        meals = int(request.form['meals'])
        exercise = request.form['exercise']
        goal = request.form['goal']

        bmi, health_status = predict_health(age, gender, height, weight, meals, exercise, goal)
        diet_plan, carbs, proteins, fats = get_meal_plan(goal)
        pie_chart = generate_pie_chart(carbs, proteins, fats)

        return render_template('result.html', bmi=bmi, health=health_status, plan=diet_plan, chart=pie_chart,
                               carbs=carbs, proteins=proteins, fats=fats)
    except Exception as e:
        print("⚠️ ERROR:", str(e))
        return "Internal Server Error: " + str(e), 500


class ServerThread(Thread):
    def __init__(self, app):
        Thread.__init__(self)
        self.server = make_server('127.0.0.1', 5000, app)
        self.ctx = app.app_context()
        self.ctx.push()

    def run(self):
        self.server.serve_forever()

    def shutdown(self):
        self.server.shutdown()

if __name__ == '__main__':
    server = ServerThread(app)
    server.start()
    time.sleep(1)
    webbrowser.open('http://127.0.0.1:5000')
