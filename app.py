from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os
from flask_cors import CORS
#flask --app app run --debug to run

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model and scaler once
lr = joblib.load('logistic_model.pkl')
scaler = joblib.load('scaler.pkl')
#rf=joblib.load('random_forest_model.pkl')


@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/form", methods=["GET"])
def form():
    return render_template("form.html")
    
@app.route("/churn", methods=["POST"])
def churn():
    try:
        global churn_val
        # Parse form data
        age = int(request.form.get("age"))
        balance = float(request.form.get("balance"))
        gender = request.form.get("gender")
        num_products = int(request.form.get("num-products"))
        active = request.form.get("active")
        model = request.form.get("model")
        
        # Convert gender and active to binary
        gender_male = 1 if gender.lower() == "male" else 0
        binary_active = 1 if active.lower() == "yes" else 0

        # Predict churn probability
        if model == "Logistic Regression":
            churn = calculateChurn_LogReg(age, balance, binary_active, num_products, gender_male)
        elif model == "Random Forest":
            churn = calculateChurn_RF(age, balance, binary_active, num_products, gender_male)
        elif model == "Extreme Gradient Boosting":
            churn = calculateChurn_XGB(age, balance, binary_active, num_products, gender_male)
        else:
            return "Invalid model selected. Please choose a valid model.", 400  # Return a 400 Bad Request

        churn_val = churn
        # Render result template
        return render_template("churn.html", churn=f"{churn:.2f}")
    except Exception as e:
        return f"An error occurred: {str(e)}"

@app.route("/recommendations")
def recommendations():
    if churn_val < 25.0:
        return render_template("recomm.html", recomm="Send a thank-you email.")
    elif churn_val < 50.0:
        return render_template("recomm.html", recomm="Send a discount coupon.")
    elif churn_val < 75.0:
        return render_template("recomm.html", recomm="Call the customer to check in.")
    else:
        return render_template("recomm.html", recomm="Provide free support for an issue.")


def calculateChurn_LogReg(age, bal, binary_active, prod, gender_Male):
    # Ensure the dataset exists
    if not os.path.exists("Modified_Churn_Modelling.csv"):
        raise FileNotFoundError("Churn dataset not found.")
    
    # Load the dataset
    df = pd.read_csv("Modified_Churn_Modelling.csv")

    # x includes all columns except 'Exited'
    x = df.drop(columns=['Exited'])
    y = df['Exited']

    # One-hot Encoding
    x = pd.get_dummies(x, drop_first=True)

    # Drop location-based features before feature selection
    x = x.drop(columns=[col for col in x.columns if 'Geography_' in col])

    # Create the input vector based on selected features
    l = [[age, bal, binary_active, prod, gender_Male]]
    
    # Scaling the input data based on the scaler used in training
    l_scaled = scaler.transform(l)

    # Predicting probability of churn
    res = lr.predict_proba(l_scaled)[:, 1]  # Probability of churning

    # Returning the churn probability as a percentage
    return res[0] * 100  # Churn probability as a percentage



def calculateChurn_RF(age, bal, binary_active, prod, gender_Male):
    # Ensure the dataset exists
    if not os.path.exists("Modified_Churn_Modelling.csv"):
        raise FileNotFoundError("Churn dataset not found.")
    
    # Load the dataset
    df = pd.read_csv("Modified_Churn_Modelling.csv")

    # x includes all columns except 'Exited'
    x = df.drop(columns=['Exited'])
    y = df['Exited']

    # One-hot Encoding
    x = pd.get_dummies(x, drop_first=True)

    # Drop location-based features before feature selection
    x = x.drop(columns=[col for col in x.columns if 'Geography_' in col])

    # Load the trained Random Forest model and scaler
    rf = joblib.load('random_forest_model.pkl')
    scaler1=joblib.load('scaler_1.pkl')

    # Create the input vector based on selected features
    l = [[age, bal, binary_active, prod, gender_Male]]
    
    # Scaling the input data based on the scaler used in training
    l_scaled = scaler1.transform(l)

    # Predicting probability of churn
    res = rf.predict_proba(l_scaled)[:, 1]  # Probability of churning

    # Returning the churn probability as a percentage
    return res[0] * 100  # Churn probability as a percentage


def calculateChurn_XGB(age, bal, binary_active, prod, gender_Male):
    # Ensure the dataset exists
    if not os.path.exists("Modified_Churn_Modelling.csv"):
        raise FileNotFoundError("Churn dataset not found.")
    
    # Load the dataset
    df = pd.read_csv("Modified_Churn_Modelling.csv")

    # x includes all columns except 'Exited'
    x = df.drop(columns=['Exited'])
    y = df['Exited']

    # One-hot Encoding
    x = pd.get_dummies(x, drop_first=True)

    # Drop location-based features before feature selection
    x = x.drop(columns=[col for col in x.columns if 'Geography_' in col])

    # Load the trained XGBoost model and scaler
    xgb = joblib.load('xgboost_model.pkl')
    scaler_xgb = joblib.load('scaler_xgb.pkl')

    # Create the input vector based on selected features
    l = [[age, bal, binary_active, prod, gender_Male]]
    
    # Scaling the input data based on the scaler used in training
    l_scaled = scaler_xgb.transform(l)

    # Predicting probability of churn
    res = xgb.predict_proba(l_scaled)[:, 1]  # Probability of churning

    # Returning the churn probability as a percentage
    return res[0] * 100  # Churn probability as a percentage


if __name__ == '__main__':
    app.run(debug=True)