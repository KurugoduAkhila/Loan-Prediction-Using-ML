import pandas as pd
from sklearn.impute import SimpleImputer

# Load the dataset
data = pd.read_csv('loan_data_set.csv')

# Display the first few rows of the dataset
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Separate features and target variable
X = data.drop('Loan_Status', axis=1)
y = data['Loan_Status']

# Handling missing values
# For categorical columns, we'll use the most frequent value to impute missing values
categorical_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
categorical_imputer = SimpleImputer(strategy='most_frequent')
X[categorical_features] = categorical_imputer.fit_transform(X[categorical_features])

# For numerical columns, we'll use the median value to impute missing values
numerical_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
numerical_imputer = SimpleImputer(strategy='median')
X[numerical_features] = numerical_imputer.fit_transform(X[numerical_features])

# Verify that there are no more missing values
print(X.isnull().sum())

# Save the cleaned data if needed
cleaned_data = pd.concat([X, y], axis=1)
cleaned_data.to_csv('cleaned_data_file.csv', index=False)

print("Data preprocessing completed and saved to 'cleaned_data_file.csv'")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = 'cleaned_data_file.csv'  # Ensure this path is correct
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"File not found: {file_path}")
    exit()

# Display first few rows and data types to debug
print("First few rows of the dataset:")
print(df.head())
print("\nData types of each column:")
print(df.dtypes)

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Clean and preprocess the data
# Convert categorical features to numeric
df['Gender'] = df['Gender'].apply(lambda x: 1 if x == 'Male' else 0)
df['Married'] = df['Married'].apply(lambda x: 1 if x == 'Yes' else 0)
df['Education'] = df['Education'].apply(lambda x: 1 if x == 'Graduate' else 0)
df['Self_Employed'] = df['Self_Employed'].apply(lambda x: 1 if x == 'Yes' else 0)
df['Property_Area'] = df['Property_Area'].map({'Urban': 2, 'Semiurban': 1, 'Rural': 0})

# Convert 'Dependents' to numeric
df['Dependents'] = df['Dependents'].replace({'3+': 3}).astype(float)

# Fill missing values with 0 (or use another strategy depending on your data)
df.fillna(0, inplace=True)

# Define features and target variable
X = df[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome',
        'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area']]
y = df['Loan_Status'].apply(lambda x: 1 if x == 'Y' else 0)

# Check feature and target shapes
print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
try:
    model.fit(X_train, y_train)
except ValueError as e:
    print(f"Error during model fitting: {e}")
    exit()

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Predict function
def predict_loan_approval():
    print("Enter the following details:")
    gender = input("Gender (Male/Female): ")
    married = input("Married (Yes/No): ")
    dependents = input("Dependents (0/1/2/3+): ")
    education = input("Education (Graduate/Not Graduate): ")
    self_employed = input("Self Employed (Yes/No): ")
    applicant_income = float(input("Applicant Income: "))
    coapplicant_income = float(input("Coapplicant Income: "))
    loan_amount = float(input("Loan Amount: "))
    loan_term = float(input("Loan Amount Term (in months): "))
    credit_history = float(input("Credit History (1 for good, 0 for bad): "))
    property_area = input("Property Area (Urban/Semiurban/Rural): ")

    # Convert categorical features to numeric
    gender = 1 if gender == 'Male' else 0
    married = 1 if married == 'Yes' else 0
    education = 1 if education == 'Graduate' else 0
    self_employed = 1 if self_employed == 'Yes' else 0
    property_area = {'Urban': 2, 'Semiurban': 1, 'Rural': 0}.get(property_area, 0)
    
    # Convert dependents input
    dependents = int(dependents.replace('+', '')) if dependents != '3+' else 3
    
    # Create a DataFrame for the input
    input_data = pd.DataFrame([[gender, married, dependents, education, self_employed,
                                applicant_income, coapplicant_income, loan_amount,
                                loan_term, credit_history, property_area]],
                              columns=['Gender', 'Married', 'Dependents', 'Education',
                                       'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome',
                                       'LoanAmount', 'Loan_Amount_Term', 'Credit_History',
                                       'Property_Area'])
    
    # Standardize the input features
    input_data_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_data_scaled)
    
    # Output the result
    if prediction[0] == 1:
        print("Loan Approved")
    else:
        print("Loan Rejected")

# Call the function to make predictions
predict_loan_approval()
