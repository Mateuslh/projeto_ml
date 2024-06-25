import os
import joblib
import unicodedata
import pandas as pd
import nltk
import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# Carregar modelo, vetorizador TF-IDF e LabelEncoder
def load_artifacts(model_path, tfidf_path, label_encoder_path):
    model = joblib.load(model_path)
    tfidf = joblib.load(tfidf_path)
    label_encoder = joblib.load(label_encoder_path)
    return model, tfidf, label_encoder

def predict_label(texto, model, tfidf, label_encoder):
    # Pré-processamento do texto
    texto_processado = proc_texto(texto)
    
    # Transformar texto em vetor TF-IDF
    texto_vetorizado = tfidf.transform([texto_processado])
    
    # Fazer a predição usando o modelo carregado
    prediction = model.predict(texto_vetorizado)
    
    # Decodificar o resultado usando o LabelEncoder
    label = label_encoder.inverse_transform(prediction)
    
    return label[0]

# Função para pré-processamento do texto
def proc_texto(texto):
    texto = unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('ascii')
    texto = texto.lower()
    texto = re.sub(r'[^\w\s]', '', texto)
    texto = texto.split()

    stplist = stopwords.words('english')
    stplist = [word.encode('ascii', 'ignore').decode('ascii') for word in stplist]
    stplist = [word.lower() for word in stplist]

    lemmatizer = WordNetLemmatizer()  # Add this line to define the lemmatizer
    texto = [lemmatizer.lemmatize(palavra) for palavra in texto if palavra not in stplist]
    texto = [palavra for palavra in texto if len(palavra) > 2]
    texto = [palavra for palavra in texto if len(palavra) < 15]

    return ' '.join(texto)

# Caminhos dos artefatos
artifacts_dir = os.path.join(os.getcwd(), 'artifacts')
model_path = os.path.join(artifacts_dir, 'model.pkl')
tfidf_path = os.path.join(artifacts_dir, 'tfidf.pkl')
label_encoder_path = os.path.join(artifacts_dir, 'label_encoder.pkl')

# Carregar modelo e artefatos
model, tfidf, label_encoder = load_artifacts(model_path, tfidf_path, label_encoder_path)
