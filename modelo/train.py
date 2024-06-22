import os
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import unicodedata
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split, GridSearchCV

nltk.download('stopwords')
nltk.download('wordnet')

def make_model():
    # Determinar o diretório base do projeto
    current_dir = os.getcwd()
    # Caminho relativo para o arquivo CSV dentro da pasta 'data'
    data_path = os.path.join(current_dir, 'data', 'fake_and_real_news.csv')

    # Carregar os dados do CSV
    data = pd.read_csv(data_path)

    print(data.isnull().sum())
    
    lemmatizer = WordNetLemmatizer()
    
    # Função para pré-processamento do texto
    def proc_texto(texto):
        texto = unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('ascii')
        texto = texto.lower()
        texto = re.sub(r'[^\w\s]', '', texto)
        texto = texto.split()

        stplist = stopwords.words('english')
        stplist = [word.encode('ascii', 'ignore').decode('ascii') for word in stplist]
        stplist = [word.lower() for word in stplist]

        texto = [lemmatizer.lemmatize(palavra) for palavra in texto if palavra not in stplist]
        texto = [palavra for palavra in texto if len(palavra) > 2]
        texto = [palavra for palavra in texto if len(palavra) < 15]

        return ' '.join(texto)

    # Aplicar pré-processamento ao texto
    data['Trata_texto'] = data['Text'].apply(lambda x: proc_texto(x))
    dados_tratados = data['Trata_texto'].values.tolist()

    # Extrair features TF-IDF com n-grams
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
    x = tfidf.fit_transform(dados_tratados).toarray()

    # Codificar a variável de saída
    le = LabelEncoder()
    y = le.fit_transform(data['label'])

    # Dividir dados em conjunto de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Inicializar e treinar modelo com hyperparameter tuning
    rf_model = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_features': ['sqrt', 'log2', None],
        'max_depth': [10, 20, 30, None]
    }
    
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_rf_model = grid_search.best_estimator_

    # Avaliar modelo
    y_pred = best_rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    print(f'Accuracy: {accuracy}')
    print(f'ROC AUC: {roc_auc}')
    print(f'F1 Score: {f1}')
    print(f'Recall: {recall}')
    print(f'Precision: {precision}')

    # Salvar modelo e artefatos necessários
    artifacts_dir = os.path.join(current_dir, 'artifacts')
    os.makedirs(artifacts_dir, exist_ok=True)

    model_path = os.path.join(artifacts_dir, 'model.pkl')
    tfidf_path = os.path.join(artifacts_dir, 'tfidf.pkl')
    label_encoder_path = os.path.join(artifacts_dir, 'label_encoder.pkl')

    joblib.dump(best_rf_model, model_path)
    joblib.dump(tfidf, tfidf_path)
    joblib.dump(le, label_encoder_path)

    print("Modelo treinado e artefatos salvos com sucesso.")

# Chamar a função para criar e salvar o modelo
make_model()
