import numpy as np
from numpy import float64, int64
import streamlit as st
import pandas as pd
import joblib
from PIL import Image
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner
import json
import requests

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def app():
    st.title('Heart Diesease Detection')
    lottie_boom = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_czuo1i1a.json")
    st_lottie(lottie_boom)
   
    model = joblib.load(open('Heart.joblib','rb'))
   
    age = st.slider('Your age:', 0, 130, 25)
    RestingBP = st.slider('Your Resting BP:', 80, 200, 25) 
    Cholesterol = st.slider('Your Cholesterol level:', 80, 200, 25) 
    FastingBS = st.multiselect('Your FastingBS:', (0, 1))
    MaxHR = st.slider('Your Maximum Heart Rate:', 50, 400, 25) 
    Oldpeak = st.slider('Oldpeak:', 0, 5,20)
    col1,col2,col3 = st.columns(3)

    with col1:
       Sex = st.multiselect('Your Gender:', ('male', 'female'))
    with col2:
       ChestPainType = st.multiselect('ChestPain Occurs?:', ('ATA', 'NAP')) 
    with col3:
       RestingECG = st.multiselect('Your Resting ECG:', ('Normal', 'ST'))
    
    col4,col5 = st.columns(2)

    with col4:
       ExerciseAngina = st.multiselect('Exercise Angina:', (0, 1))
    with col5:
       ST_Slope = st.multiselect('Normal heart?:', ('Up', 'Flat'))
   
    final_df = pd.DataFrame({'Age':[age],'Sex':[Sex],'ChestPainType':[ChestPainType],'RestingBP':[RestingBP],
    'Cholesterol':[Cholesterol],'FastingBS':[FastingBS],'RestingECG':[RestingECG],'MaxHR':[MaxHR],
    'ExerciseAngina':[ExerciseAngina],'Oldpeak':[Oldpeak], 'ST_Slope':[ST_Slope]})
   
    final_df['FastingBS'] = final_df['FastingBS'].apply(lambda x: 1 if x == 1 else 0)
    final_df['Sex'] = final_df['Sex'].apply(lambda x: 'male' if x == 'male' else 'female')
    final_df['ChestPainType'] = final_df['ChestPainType'].apply(lambda x: 'ATA' if x == 'ATA' else 'NAP')
    final_df['RestingECG'] = final_df['RestingECG'].apply(lambda x: 'Normal' if x == 'Normal' else 'ST')
    final_df['ExerciseAngina'] = final_df['ExerciseAngina'].apply(lambda x: 1 if x == 1 else 0) 
    final_df['ST_Slope'] = final_df['ST_Slope'].apply(lambda x: 'Up' if x == 'Up' else 'Flat')
   
    print(final_df.info())
    print(final_df.head())
    print(final_df['FastingBS'].dtype)
    
    
    #final_df['ChestPainType']=final_df['ChestPainType'][0]
    #final_df['RestingECG']=final_df['RestingECG'][0]
    #final_df['ExerciseAngina']=final_df['ExerciseAngina'][0]
    #final_df['ST_Slope']=final_df['ST_Slope'][0]
    #final_df['Age'] =  final_df['Age'].astype(int64)
    final_df['Sex'] =  final_df['Sex'].astype('string')
    final_df['ChestPainType'] =  final_df['ChestPainType'].astype('string')
    #final_df['RestingBP'] =  final_df['RestingBP'].astype(int64)
    #final_df['Cholesterol'] =  final_df['Cholesterol'].astype(int64)
    #final_df['FastingBS'] =  final_df['FastingBS'].astype(int64)
    final_df['RestingECG'] =  final_df['RestingECG'].astype('string')
    #final_df['MaxHR'] =  final_df['MaxHR'].astype(int64)
    final_df['ExerciseAngina'] =  final_df['ExerciseAngina'].astype(str)
    final_df['Oldpeak'] =  final_df['Oldpeak'].astype(float64)
    final_df['ST_Slope'] =  final_df['ST_Slope'].astype('string')

    if st.button('Predict'):
       lottie_spinner = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_xah4ixac.json")
       with st_lottie_spinner(lottie_spinner, key="Wait"):    
         predictions = model.predict(final_df)
         if predictions == 1:
               st.header("Consider visiting a doctor. Your heart maybe at risk!!!")
         else:
               st.header("You are absolutely fine. Still visit a doctor for verification!")   