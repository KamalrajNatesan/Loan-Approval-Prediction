# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("dataset.csv")

# Identify numeric columns
numeric_columns = data.select_dtypes(include=['number']).columns

# Fill missing values only for numeric columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median(), inplace=False)

# Encode categorical variables
encoder = LabelEncoder()
categorical_columns = ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]
for column in categorical_columns:
    data[column] = encoder.fit_transform(data[column])

# Split the data into features (X) and the target variable (y)
X = data.drop("loan_status", axis=1)
y = data["loan_status"]

# Standardize the input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_scaled, y)

# Streamlit app
st.title("Loan Approval Prediction")
# Sidebar with input features
st.sidebar.header("Input Features")
new_data = {}
new_data["person_age"] = st.sidebar.slider("Person Age", min_value=18, max_value=100, value=25)
new_data["person_income"] = st.sidebar.slider("Person Income", min_value=0, max_value=500000, value=10000)
new_data["person_emp_length"] = st.sidebar.slider("Person Employment Length", min_value=0, max_value=50, value=5)
new_data["loan_amnt"] = st.sidebar.slider("Loan Amount", min_value=0, max_value=50000, value=10000)
new_data["loan_int_rate"] = st.sidebar.slider("Loan Interest Rate", min_value=0.0, max_value=30.0, value=7.5)
new_data["loan_percent_income"] = st.sidebar.slider("Loan Percent Income", min_value=0, max_value=100, value=16)
new_data["cb_person_cred_hist_length"] = st.sidebar.slider("Credit History Length", min_value=0, max_value=50, value=7)

# Map categorical features to numeric values
home_ownership_mapping = {"MORTGAGE": 0, "OTHER": 1, "OWN": 2, "RENT": 3}
loan_intent_mapping = {"DEBT_CONSOLIDATION": 0, "EDUCATION": 1, "HOME_IMPROVEMENT": 2, "MEDICAL": 3, "PERSONAL": 4,
                       "VENTURE": 5}
loan_grade_mapping = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
default_on_file_mapping = {"N": 0, "Y": 1}

new_data["person_home_ownership"] = st.sidebar.selectbox("Home Ownership", list(home_ownership_mapping.keys()))
new_data["loan_intent"] = st.sidebar.selectbox("Loan Intent", list(loan_intent_mapping.keys()))
new_data["loan_grade"] = st.sidebar.selectbox("Loan Grade", list(loan_grade_mapping.keys()))
new_data["cb_person_default_on_file"] = st.sidebar.selectbox("Default on File", list(default_on_file_mapping.keys()))

# Convert categorical selections to numeric using the mapping
new_data["person_home_ownership"] = home_ownership_mapping[new_data["person_home_ownership"]]
new_data["loan_intent"] = loan_intent_mapping[new_data["loan_intent"]]
new_data["loan_grade"] = loan_grade_mapping[new_data["loan_grade"]]
new_data["cb_person_default_on_file"] = default_on_file_mapping[new_data["cb_person_default_on_file"]]

# Ensure the feature names in new_data match the original training data (X)
new_data = pd.DataFrame([new_data], columns=X.columns)

# Use the Random Forest model for prediction
new_data_scaled = scaler.transform(new_data)
new_data_scaled = pd.DataFrame(new_data_scaled, columns=X.columns)
prediction = rf_model.predict(new_data_scaled)

# Display prediction result
st.header("Prediction Result")
st.subheader("Loan Status:")
if prediction[0] == 0:
    st.success("Approved - The user will pay the loan amount.")
else:
    st.error("Not Approved - The user will not pay the loan amount.")

