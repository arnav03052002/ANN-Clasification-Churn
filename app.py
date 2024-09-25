import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Loading the traied model
model=tf.keras.models.load_model('model.h5')

# Importing encoders and scalers

with open('label_encoder_generator.pkl','rb') as file:
    encoder_gender=pickle.load(file)

with open('ohe.pkl','rb') as file:
    ohe_geo=pickle.load(file)

with open('scale.pkl','rb') as file:
    scaler=pickle.load(file)

# Streamlit app

st.title('Customer churn prediction')

geography=st.selectbox('Geography', ohe_geo.categories_[0])
gender=st.selectbox('Gender', encoder_gender.classes_)
age=st.slider('Age',18,100)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_salary=st.number_input('Estimated Salary')
tenure=st.slider('Tenure',0,10)
num_of_products=st.slider('Number of product',0,4)
has_cr_card=st.selectbox('Has credit card',[0,1])
is_active_member=st.selectbox('Is active Member',[0,1])

input_data=pd.DataFrame({
    'CreditScore':[credit_score],
    # 'Geography':[encoder_gender.transform([gender])[0]],
    'Gender':[encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary]
})

geo_encoded=ohe_geo.transform([[geography]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=ohe_geo.get_feature_names_out(['Geography']))

input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

# scaleing input data
input_data_scaled=scaler.transform(input_data )

prediction=model.predict(input_data_scaled)

prediction_prob=prediction[0][0]

st.write(f"Churn probability: {prediction_prob:.2f}")

if prediction_prob>0.5:
    st.write('Customer is likely to churn')
else:
    st.write('Customer is unlikely to churn')
