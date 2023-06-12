import streamlit as st
import pandas as pd 
import pickle
import joblib 
import numpy as np
from scipy.sparse import csr_matrix
import sklearn 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin 
import re 
import unicodedata


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if isinstance(X, csr_matrix):
            X = X.todense()
        df = pd.DataFrame(X)
        return df

          
def clean_text(txt):
    nfkd = unicodedata.normalize('NFKD', txt)
    palavraSemAcento = u"".join([c for c in nfkd if not unicodedata.combining(c)])
    palavraSemAcento = re.sub(r"(@[A-Za]+)|([^A-Za-z \t])|(\w+:\/\/\S+)| ^rt|http.+?","", palavraSemAcento)
    palavraSemAcento = str(palavraSemAcento).lower()
    palavraSemAcento = re.sub(r"\d+", "", palavraSemAcento)
    palavraSemAcento = re.sub(r'  ', ' ', palavraSemAcento)
    palavraSemAcento = re.sub(r'compartilheversao para impressao comentarios',' ',
                                palavraSemAcento)
    return palavraSemAcento


def preprocess_data(texto):
    pipeline = Pipeline([
        ('preprocessor', TextPreprocessor()),
        ('vectorizer', CountVectorizer(ngram_range=(1, 1))),
        ('scaler', StandardScaler(with_mean=False))
    ])
  
    # Transforma o texto usando o pipeline
    vectorized_text = pipeline['vectorizer'].fit_transform([texto])
    vectorized_text = pipeline['scaler'].fit_transform(vectorized_text)
    return vectorized_text


# Carrega o modelo
try:
    model = pickle.load(open('PIPELINE_texto_padronizado_AdaBoost.joblib', 'rb'))
except Exception as e:
    st.error(f"Erro ao carregar o modelo: {e}")


# Configuração do Streamlit
header_image ='/home/mel-iza/vaccine_fake_news/src/wallpaper_2.png'
st.image(header_image, use_column_width=True)
#st.title("Modelo de Classificação de Texto")
st.subheader("Detector de notícias falsas sobre vacinação")
texto = st.text_input("Digite o texto:")
if st.button("Analisar"):
    # Limpeza do texto
    texto_limpo = clean_text(texto)
    
    # Pré-processamento e predição
    predicao = model.predict_proba([texto_limpo])
    probabilidade = round(np.max(predicao[0]) * 100, 2)
    
    # Exibição do resultado
    st.write(f"Probabilidade: {probabilidade}%")

#trigger = st.button('Enviar', on_click=probabilidade)
#model = joblib.load('/home/mel-iza/vaccine_fake_news/PIPELINE_texto_padronizado_AdaBoost.joblib')


#input_text = clean_text(input_text)
#predicao = model.predict_proba([input_text])
#probabilidade = round(np.max(predicao[0]) * 100, 2)

#print(f"Probabilidade: {probabilidade}%")

#preprocessed_text = process_data(input_text)
#prediction = predict(preprocessed_text, model) 


