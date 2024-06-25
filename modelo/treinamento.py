import pandas as pd
import numpy as np
import re
import unicodedata
import logging
import os
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def treinar_modelo():

    current_dir = os.getcwd()
    # Verificar se os arquivos de artefatos já existem
    artifacts_dir = os.path.join(current_dir, 'artifacts')
    
    modelo_path = os.path.join(artifacts_dir, 'modelo_logistic_regression.pkl')
    vectorizer_path = os.path.join(artifacts_dir, 'vectorizer.pkl')
    
    if os.path.exists(modelo_path) and os.path.exists(vectorizer_path):
        logger.info("Arquivos de artefatos já existem. Não é necessário criar o modelo novamente.")
        return
    
    nltk.download('stopwords')

    # Carregar os dados
    data = pd.read_csv('data/fake_and_real_news.csv')
    #data = data.sample(frac=0.5, random_state=42)
    data = data.sample(frac=0.3, random_state=42)

    porter_stemmer = PorterStemmer()

    # Função para pré-processamento do texto
    def proc_texto(data):
        words = re.sub(r"[^A-Za-z]"," ", data).lower().split()
        words = [porter_stemmer.stem(word) for word in words if word not in stopwords.words('english')]
        words = ' '.join(words)
        return words

    data['features'] = data['Text'].apply(proc_texto)

    X = data['features'].values
    Y = data['label'].values

    vectorizer = TfidfVectorizer()
    vectorizer.fit(X)
    X = vectorizer.transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

    lr = LogisticRegression()
    lr.fit(X_train, Y_train)

    # Salvar o modelo treinado em um arquivo .pkl
    modelo_file_path = os.path.join(artifacts_dir, 'modelo_logistic_regression.pkl')
    with open(modelo_file_path, 'wb') as file:
        pickle.dump(lr, file)
        
    # Salvar o vetor de TF-IDF em um arquivo .pkl
    vectorizer_file_path = os.path.join(artifacts_dir, 'vectorizer.pkl')
    with open(vectorizer_file_path, 'wb') as file:
        pickle.dump(vectorizer, file)

    # Prever nos dados de teste e calcular a acurácia de teste
    test_pred = lr.predict(X_test)
    test_acc = accuracy_score(Y_test, test_pred)
    print(f'Acurácia de Teste: {test_acc*100:.2f}%')

    # Outras métricas para os dados de teste
    print("\nMétricas de Teste:")
    print(classification_report(Y_test, test_pred))

    # Matriz de confusão para os dados de teste
    print("\nMatriz de Confusão (Teste):")
    print(confusion_matrix(Y_test, test_pred))

treinar_modelo()
