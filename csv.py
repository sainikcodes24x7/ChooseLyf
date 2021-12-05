import streamlit as st
import pandas as pd
from PIL import Image

data = pd.read_csv('Book.csv')
df=data.T

st.title('Results')
#image = Image.open('.jpg')
#st.image(image, width=500)
st.write(df.astype(str))
#if data{}
st.write('Thank you for consulting us.')
