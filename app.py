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
import sys

sys.path.insert(1, '..')

class SparseToDataFrameTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return pd.DataFrame(X.todense())

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
        ('to_dataframe', SparseToDataFrameTransformer()),
        ('scaler', StandardScaler(with_mean=True))
    ])
  
    # Transforma o texto usando o pipeline
    vectorized_text = pipeline['vectorizer'].fit_transform([texto])
    vectorized_text = pipeline['scaler'].fit_transform(vectorized_text)
    vectorized_text = pipeline['to_dataframe'].fit_transform(vectorized_text)
    
    return vectorized_text


# Carrega o modelo
try:
    model = pickle.load(open('model/pickle_texto_padronizadoAdaBoost.pkl', 'rb'))
except Exception as e:
    st.error(f"Erro ao carregar o modelo: {e}")

def get_prediction(texto, model):
    texto_limpo = clean_text(texto)
    predicao = model.predict_proba([texto_limpo])
    probabilidade = round(np.max(predicao[0]) * 100, 2)
    return probabilidade

def get_response(probabilidade):
    if 0.5 <= probabilidade < 0.53:
        st.error(f'Essa informação sobre vacina tem {probabilidade}% de probabilidade de ser falsa...')
    elif 0.53 <= probabilidade < 0.56:
        st.warning(f'Hum, não tenho muita certeza sobre essa informação. Tem {probabilidade}% de probabilidade de ser verdadeira...')
    elif probabilidade >= 0.56:
        st.success(f'Essa informação sobre vacina tem {probabilidade}% de probabilidade de ser verdadeira...')
    else:
        st.error(f'A probabilidade ({probabilidade}) está fora do intervalo esperado.')

header_image = 'src/wallpaper_2.png'
st.image(header_image, use_column_width=True)

st.subheader("Detector de notícias falsas sobre vacinação")
texto = st.text_input("Digite um texto relacionado aos temas sobre vacinas, vacinação, imunização, imunizantes")
st.divider()

if st.button("Enviar"):
    if texto.strip() == "":
        st.error("Por favor, digite um texto válido.")
    else:
        with st.spinner("Analisando..."):
            probabilidade = get_prediction(texto, model)
            get_response(probabilidade)
            

    


