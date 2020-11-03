# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 22:25:02 2020

@author: APPU
"""


import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from PIL import Image
import streamlit as st


image=Image.open('index.jpg')
st.image(image,caption=' ',use_column_width=True)

st.write("""
## Diabetes Detection using Machine Learning
""")

df= pd.read_csv("diabetes.csv")

st.subheader('Training Data: ')
st.dataframe(df)
#st.write(df.describe())
chart=st.line_chart(df)




x=df[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI',
      'DiabetesPedigreeFunction','Age']]
y=df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)

from sklearn.ensemble import RandomForestClassifier
lm = RandomForestClassifier()


def user_input():
    Pregnancies=st.sidebar.slider('Pregnancies: ',0,20,2)
    Glucose=st.sidebar.slider('Glucose: ',40,240,120)
    BloodPressure=st.sidebar.slider('Blood Pressure: ',20,200,60)
    SkinThickness=st.sidebar.slider('SkinThickness: ',0,100,20)
    Insulin=st.sidebar.slider('Insulin: ',10,900,35)
    BMI=st.sidebar.slider('BMI: ',10,70,30)
    DiabetesPedigreeFunction=st.sidebar.slider('DiabetesPedigreeFunction: ',0.078, 2.42, 0.3725)
    Age=st.sidebar.slider('Age: ',20,90,30)
    
    
    input_data={'Pregnancies': Pregnancies,'Glucose': Glucose, 'Blood Pressure': BloodPressure,
    'Skin Thickness': SkinThickness,'Insulin': Insulin,'BMI': BMI,
    'DPF': DiabetesPedigreeFunction,'Age': Age}
    
    features=pd.DataFrame(input_data,index=[0])
    return features 

get_user_input=user_input()

st.subheader('USer Input: ')
st.write(get_user_input)

lm.fit(X_train, y_train)

st.subheader('Model Accuracy: ')
#st.write( str(accuracy_score(y_test, lm.predict(X_test)) * 100) + '%' )
st.write(accuracy_score(y_test, lm.predict(X_test).round()))
prediction=lm.predict(get_user_input)

st.subheader('Classification: ')
st.write(prediction)

