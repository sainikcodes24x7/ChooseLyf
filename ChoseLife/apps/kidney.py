import numpy as np
from numpy import float64, int64
import streamlit as st
import pandas as pd
import joblib
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
    st.title('Kidney Diesease Detection')
    lottie_boom = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_zvkbowj2.json")
    st_lottie(lottie_boom)
   
    model = joblib.load(open('KIdney.joblib','rb'))
   
    age = st.number_input('Insert your age')
    bp = st.number_input('Enter Blood Pressure')
    sg = st.number_input('Enter Urine Specific Gravity')
    al = st.number_input('Enter albumin level')
    su = st.number_input('Your Sugar level') 
    
    col1,col2,col3,col4 = st.columns(4)

    with col1:
       rbc = st.multiselect('Your RBC type:', ('normal', 'abnormal'))
    with col2:
       pc = st.multiselect('ChestPain Occurs?:', ('normal', 'abnormal')) 
    with col3:
       pcc = st.multiselect('Pus Cell Clumps:', ('notpresent', 'present'))
    with col3:
       ba = st.multiselect('Your Resting ECG:', ('notpresent', 'present'))   
    
    bgr = st.number_input('Insert your blood glucose level')
    bu = st.number_input('Insert your blood urea level')
    sc = st.number_input('Insert your serum creatinine level')
    sod = st.number_input('Insert your sodium level')
    pot = st.number_input('Insert your potassium level')
    hemo = st.number_input('Insert your hemoglobin level')

    col5,col6,col7 = st.columns(3)

    with col5:
       htn = st.multiselect('High blood pressure?', ('yes', 'no'))
    with col6:
       dm = st.multiselect('suffering from Diabetes?', ('yes', 'no'))
    with col7:
       cad = st.multiselect('Coronary Artery Disease present?', ('yes', 'no')) 

    col8,col9,col10 = st.columns(3)

    with col8:
       appet = st.multiselect('Your apetite:', ('good', 'poor'))
    with col9:
       pe = st.multiselect('Pedal Edema:', ('yes', 'no'))
    with col10:
       ane = st.multiselect('Anemia', ('yes', 'no')) 

    final_df = pd.DataFrame({'age':[age],'bp':[bp],'sg':[sg],'al':[al],
    'su':[su],'rbc':[rbc],'pc':[pc],'pcc':[pcc],'ba':[ba],'bgr':[bgr], 'bu':[bu], 'sc':[sc], 'sod':[sod],
    'pot':[pot], 'hemo':[hemo], 'htn':[htn], 'dm':[dm], 'cad':[cad], 'appet':[appet], 'pe':[pe], 'ane':[ane]})
   
    final_df['rbc'] = final_df['rbc'].apply(lambda x: 'normal' if x == 'normal' else 'abnormal')
    final_df['pc'] = final_df['pc'].apply(lambda x: 'normal' if x == 'normal' else 'abnormal')
    final_df['pcc'] = final_df['pcc'].apply(lambda x: 'present' if x == 'present' else 'notpresent')
    final_df['ba'] = final_df['ba'].apply(lambda x: 'present' if x == 'present' else 'notpresent')
    final_df['htn'] = final_df['htn'].apply(lambda x: 'yes' if x == 'yes' else 'no')
    final_df['dm'] = final_df['dm'].apply(lambda x: 'yes' if x == 'yes' else 'no')
    final_df['cad'] = final_df['cad'].apply(lambda x: 'yes' if x == 'yes' else 'no')
    final_df['appet'] = final_df['appet'].apply(lambda x: 'good' if x == 'good' else 'poor')
    final_df['pe'] = final_df['pe'].apply(lambda x: 'yes' if x == 'yes' else 'no')
    final_df['ane'] = final_df['ane'].apply(lambda x: 'yes' if x == 'yes' else 'no')
   
    #print(final_df.info())
    #print(final_df.head())
    
    
    final_df['age'] =  final_df['age'].astype(float64)
    final_df['bp'] =  final_df['bp'].astype(float64)
    final_df['sg'] =  final_df['sg'].astype(float64)
    final_df['al'] =  final_df['al'].astype(float64)
    final_df['su'] =  final_df['su'].astype(float64)
    final_df['bgr'] =  final_df['bgr'].astype(float64)
    final_df['bu'] =  final_df['bu'].astype(float64)
    final_df['sc'] =  final_df['sc'].astype(float64)
    final_df['sod'] =  final_df['sod'].astype(float64)
    final_df['pot'] =  final_df['pot'].astype(float64)
    final_df['hemo'] =  final_df['hemo'].astype(float64)

    if st.button('Predict'): 
       lottie_spinner = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_xah4ixac.json")
       with st_lottie_spinner(lottie_spinner, key="Wait"):    
         predictions = model.predict(final_df)
         if predictions == 1:
               st.header("Consider visiting a doctor. Your kidney maybe at risk!!!")
         else:
               st.header("You are absolutely fine. Still visit a doctor for verification!") 