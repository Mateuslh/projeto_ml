from flask import Flask, request, jsonify
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Carregar o modelo treinado
with open('artifacts/modelo_logistic_regression.pkl', 'rb') as file:
    modelo = pickle.load(file)

# Carregar o vetor de TF-IDF
with open('artifacts/vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

porter_stemmer = PorterStemmer()
nltk.download('stopwords')

# Função para pré-processamento do texto
def proc_texto(data):
    words = re.sub(r"[^A-Za-z]"," ", data).lower().split()
    words = [porter_stemmer.stem(word) for word in words if word not in stopwords.words('english')]
    words = ' '.join(words)
    return words

def predict(noticia):
        
    noticia_proc = proc_texto(noticia)
    noticia_vec = vectorizer.transform([noticia_proc])
    predicao = modelo.predict(noticia_vec)
    
    resultado = "Falsa" if predicao[0] == 'Fake' else "Verdadeira"
    
    return resultado
