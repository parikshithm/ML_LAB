"""
Application that predicts laptop prices based on user input fields.
"""

import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

# Create Flask app
app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

# Load the trained pipeline
#with open('C:/Users/moudg/Desktop/great_lakes_docs/TERM 5/Predictive analytics lab/INCLASS ASSIGNMENT 1/model.pkl', 'rb') as f:
#    pipeline = pickle.load(f)
    
# Define the categorical features and their possible values
categorical_columns = ['Company', 'TypeName', 'Cpu Name', 'OpSys', 'Gpu']
feature_order = ['Company', 'TypeName', 'Inches', 'Ram', 'Gpu', 'OpSys', 'Touchscreen', 'Cpu Name']

    
def preprocess_input(user_input):
    """
    Preprocess form data to match the model's expected input.
    """
    # Convert form data to a DataFrame
    df = pd.DataFrame([user_input])
    
    # Apply one-hot encoding
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    
    # Ensure all expected columns are present, filling missing ones with 0
    for col in model.feature_names_in_:
        if col not in df.columns:
            df[col] = 0

    # Align the columns with the model's expected feature order
    df = df[model.feature_names_in_]

    return df.astype(float)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form data
        user_input = {
            "Company": request.form.get("Company"),
            "TypeName": request.form.get("TypeName"),
            "Inches": request.form.get("Inches"),
            "Ram": int(request.form.get("Ram")),
            "gpu": request.form.get("gpu"),
            "os": request.form.get("os"),
            "touchscreen": int(request.form.get("touchscreen")),
            "cpu": int(request.form.get("cpu")),
        }

        # Preprocess the input
        processed_data = preprocess_input(user_input)

        # Make prediction
        prediction = model.predict(processed_data)[0]

        # Return prediction to the template
        return render_template('index.html', prediction_text=f'Predicted Price: {round(prediction, 2)}')
    
    except Exception as e:
        # Log and handle the error
        print(e)
        return render_template('index.html', prediction_text="An error occurred during prediction. Please try again.")

'''
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        company = request.form['company']
        typename = request.form['type']
        inches = float(request.form['inches'])
        ram = int(request.form['ram'])
        gpu = request.form['gpu']
        os = request.form['os']
        touchscreen = int(request.form['touchscreen'])
        cpu_name = request.form['cpu']

        # Create input DataFrame
        input_data = pd.DataFrame({
            'Company': [company],
            'TypeName': [typename],
            'Inches': [inches],
            'Ram': [ram],
            'Gpu': [gpu],
            'OpSys': [os],
            'Touchscreen': [touchscreen],
            'Cpu Name': [cpu_name]
        })

        # Make prediction
        predicted_price_log = pipeline.predict(input_data)
        predicted_price = np.exp(predicted_price_log)  # Convert log-price back to original scale

        return render_template('index.html', prediction_text=f'Predicted Laptop Price: ${predicted_price[0]:.2f}')
    except Exception as e:
        return f"Error occurred: {str(e)}"
'''
if __name__ == "__main__":
    app.run(debug=True)
