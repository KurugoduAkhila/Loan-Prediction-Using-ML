from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Load and prepare the model
df = pd.read_csv('cleaned_data_file.csv')
df['Gender'] = df['Gender'].apply(lambda x: 1 if x == 'Male' else 0)
df['Married'] = df['Married'].apply(lambda x: 1 if x == 'Yes' else 0)
df['Education'] = df['Education'].apply(lambda x: 1 if x == 'Graduate' else 0)
df['Self_Employed'] = df['Self_Employed'].apply(lambda x: 1 if x == 'Yes' else 0)
df['Property_Area'] = df['Property_Area'].map({'Urban': 2, 'Semiurban': 1, 'Rural': 0})
df['Dependents'] = df['Dependents'].replace({'3+': 3}).astype(float)
df.fillna(0, inplace=True)

X = df[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 
        'Credit_History', 'Property_Area']]
y = df['Loan_Status'].apply(lambda x: 1 if x == 'Y' else 0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Check for missing fields
        required_fields = ['gender', 'married', 'dependents', 'education', 'self_employed', 
                           'applicant_income', 'coapplicant_income', 'loan_amount', 
                           'loan_term', 'credit_history', 'property_area']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({'error': f'Missing fields: {", ".join(missing_fields)}'}), 400
        
        # Convert categorical features to numeric
        gender = 1 if data['gender'] == 'Male' else 0
        married = 1 if data['married'] == 'Yes' else 0
        education = 1 if data['education'] == 'Graduate' else 0
        self_employed = 1 if data['self_employed'] == 'Yes' else 0
        property_area = {'Urban': 2, 'Semiurban': 1, 'Rural': 0}.get(data['property_area'], 0)
        
        dependents = int(data['dependents'].replace('+', '')) if data['dependents'] != '3+' else 3
        
        # No scaling down of loan amount, keep it as is
        loan_amount = float(data['loan_amount'])  # Take the entered loan amount directly

        # Create a DataFrame for the input
        input_data = pd.DataFrame([[gender, married, dependents, education, self_employed,
                                    float(data['applicant_income']),
                                    float(data['coapplicant_income']),
                                    loan_amount,  # Directly use the entered loan amount
                                    float(data['loan_term']),
                                    float(data['credit_history']),
                                    property_area]],
                                  columns=['Gender', 'Married', 'Dependents', 'Education',
                                           'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome',
                                           'LoanAmount', 'Loan_Amount_Term', 'Credit_History',
                                           'Property_Area'])
        
        # Standardize the input features
        input_data_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_data_scaled)
        
        # Output the result
        status = 'Loan Approved' if prediction[0] == 1 else 'Loan Rejected'
        return jsonify(status=status)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
