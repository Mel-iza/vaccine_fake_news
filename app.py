import streamlit as st
from utils import PrepProcesor, columns 

import numpy as np
import pandas as pd
import joblib

model = joblib.load('textopadronizado_Naive_Bayes.joblib')


st.title(' Vaccine Fake News Detection 💉🧪')
header_image ='wallpaper_2.png'
st.image(header_image, use_column_width=True)

st.text_input('Digite sua mensagem ou texto sobre vacina/vacinação')
st.sidebar.header(" ")
sidebar_information_1 = st.sidebar.selectbox("Fake News about vaccine", 
("Como saber o que é uma fake new"))
st.write(sidebar_information_1)
sidebar_information_2 = st.sidebar.selectbox("About Project", 
("Tcc", "motivations"))
st.write(sidebar_information_2)
sidebar_information_3 = st.sidebar.selectbox("Data", 
("How data was collected?", "Visualizations"))
st.write(sidebar_information_3)


def predict(): 
    row = np.array([text]) 
    X = pd.DataFrame([row], columns = columns)
    prediction = model.predict_proba(X)
    if prediction[0] > 0.50: 
        st.success('É bem possível que essa informação sobre vacina seja falsa.')
    else: 
        st.error('É bem possível que essa informação sobre vacina seja verdadeira') 

trigger = st.button('Predict', on_click=predict)