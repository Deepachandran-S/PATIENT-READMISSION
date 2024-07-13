!pip instal streamlit
!pip install sklearn
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the saved model, label encoders, and scaler
with open('lgb_model.pkl', 'rb') as file:
    model = pickle.load(file)
with open('label_encoders2.pkl', 'rb') as file:
    label_encoders = pickle.load(file)
#with open('scaler.pkl', 'rb') as file:
    #scaler = pickle.load(file)

# Define the preprocessing function for new input
def preprocess_input(input_df):
    # Encode 'age' as discrete
    age_dict = {'[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35, '[40-50)': 45, '[50-60)': 55,
                '[60-70)': 65, '[70-80)': 75, '[80-90)': 85, '[90-100)': 95}
    input_df['age'] = input_df['age'].map(age_dict)
    input_df['A1Cresult']=input_df['A1Cresult'].replace({'None':0, 'Norm':1, '>7':2, '>8':3})
    # Encoding categorical variables
    for col in label_encoders:
      try:
        input_df[col] = label_encoders[col].transform(input_df[col])
      except:
        continue


    # Standardize numeric columns
    numeric_cols = input_df.select_dtypes(include=[np.number]).columns
    input_df[numeric_cols] = StandardScaler().fit_transform(input_df[numeric_cols])
    
    return input_df

# Define the Streamlit app
st.title('Medical Readmission Prediction')

# Define input fields
race = st.selectbox('Race', ['Caucasian', 'AfricanAmerican', 'Other', 'Asian', 'Hispanic'])
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.selectbox('Age', ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'])
admission_type_id = st.selectbox('Admission Type ID', [1, 2, 3, 4, 5, 6, 7, 8])
discharge_disposition_id = st.selectbox('Discharge Disposition ID', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 30])
admission_source_id = st.selectbox('Admission Source ID', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 20, 21, 22, 25, 26])
time_in_hospital = st.number_input('Time in Hospital', min_value=1, max_value=14, value=1)
medical_specialty = st.selectbox('Medical Specialty', ['Cardiology', 'Endocrinology', 'Family/GeneralPractice', 'InternalMedicine', 'Nephrology', 'Neurology', 'ObstetricsandGynecology', 'Oncology', 'Orthopedics', 'Pediatrics', 'Pulmonology', 'Radiology', 'Surgery-General', 'Urology', 'Other'])
num_lab_procedures = st.number_input('Number of Lab Procedures', min_value=0, max_value=132, value=0)
num_procedures = st.number_input('Number of Procedures', min_value=0, max_value=6, value=0)
num_medications = st.number_input('Number of Medications', min_value=1, max_value=81, value=1)
number_outpatient = st.number_input('Number of Outpatient Visits', min_value=0, max_value=42, value=0)
number_inpatient = st.number_input('Number of Inpatient Visits', min_value=0, max_value=21, value=0)
number_diagnoses = st.number_input('Number of Diagnoses', min_value=1, max_value=16, value=1)
A1Cresult = st.selectbox('A1C Result', ['None', 'Norm', '>7', '>8'])
metformin = st.selectbox('Metformin', ['No', 'Steady', 'Up', 'Down'])
glipizide = st.selectbox('Glipizide', ['No', 'Steady', 'Up', 'Down'])
glyburide = st.selectbox('Glyburide', ['No', 'Steady', 'Up', 'Down'])
pioglitazone = st.selectbox('Pioglitazone', ['No', 'Steady', 'Up', 'Down'])
insulin = st.selectbox('Insulin', ['No', 'Steady', 'Up', 'Down'])
change = st.selectbox('Change', ['No', 'Ch'])
diabetesMed = st.selectbox('Diabetes Medication', ['No', 'Yes'])

# Create a dictionary for the input features
input_data = {
    'race': race,
    'gender': gender,
    'age': age,
    'admission_type_id': admission_type_id,
    'discharge_disposition_id': discharge_disposition_id,
    'admission_source_id': admission_source_id,
    'time_in_hospital': time_in_hospital,
    'medical_specialty': medical_specialty,
    'num_lab_procedures': num_lab_procedures,
    'num_procedures': num_procedures,
    'num_medications': num_medications,
    'number_outpatient': number_outpatient,
    'number_inpatient': number_inpatient,
    'number_diagnoses': number_diagnoses,
    'A1Cresult': A1Cresult,
    'metformin': metformin,
    'glipizide': glipizide,
    'glyburide': glyburide,
    'pioglitazone': pioglitazone,
    'insulin': insulin,
    'change': change,
    'diabetesMed': diabetesMed,
    'service_utilization': number_outpatient + number_inpatient
}

# Convert the input data to a DataFrame
input_df = pd.DataFrame([input_data])

# Preprocess the input data
input_df = preprocess_input(input_df)

# Make a prediction
if st.button('Predict'):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    # Display the result
    st.write('Prediction: ', 'Readmitted' if prediction[0].round() == 1 else 'Not Readmitted')
    st.write('Prediction Probability: ', prediction_proba[0][0],' No ', prediction_proba[0][1],' Yes ')
