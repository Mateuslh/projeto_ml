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
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
import logging

nltk.download('stopwords')
nltk.download('wordnet')

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def make_model():
    logger.info("Iniciando a criação do modelo")

    # Determinar o diretório base do projeto
    current_dir = os.getcwd()
    # Verificar se os arquivos de artefatos já existem
    artifacts_dir = os.path.join(current_dir, 'artifacts')
    if os.path.exists(artifacts_dir):
        logger.info("Arquivos de artefatos já existem. Não é necessário criar o modelo novamente.")
        return
    
    # Caminho relativo para o arquivo CSV dentro da pasta 'data'
    data_path = os.path.join(current_dir, 'data', 'fake_and_real_news.csv')

    # Carregar os dados do CSV
    data = pd.read_csv(data_path)
    data = data.sample(frac=0.1, random_state=42)
    logger.info(f"Dados carregados com sucesso. {data.shape[0]} linhas e {data.shape[1]} colunas.")

    logger.info("Verificando dados nulos...")
    print(data.isnull().sum())

    # Verificar e remover duplicações
    logger.info(f"Linhas duplicadas antes da remoção: {data.duplicated().sum()}")
    data = data.drop_duplicates()
    logger.info(f"Linhas duplicadas após a remoção: {data.duplicated().sum()}")

    lemmatizer = WordNetLemmatizer()
    
    # Função para pré-processamento do texto
    def proc_texto(texto):
        texto = unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('ascii')
        texto = texto.lower()
        texto = re.sub(r'http\S+', '', texto)  # Remover URLs
        texto = re.sub(r'@\w+', '', texto)    # Remover menções
        texto = re.sub(r'\d+', '', texto)     # Remover números
        texto = re.sub(r'[^\w\s]', '', texto) # Remover caracteres não-alfabéticos
        texto = texto.split()

        stplist = stopwords.words('english')
        stplist = [word.encode('ascii', 'ignore').decode('ascii') for word in stplist]
        stplist = [word.lower() for word in stplist]

        texto = [lemmatizer.lemmatize(palavra) for palavra in texto if palavra not in stplist]
        texto = [palavra for palavra in texto if len(palavra) > 2]
        texto = [palavra for palavra in texto if len(palavra) < 15]

        return ' '.join(texto)

    # Aplicar pré-processamento ao texto
    logger.info("Iniciando o pré-processamento dos textos...")
    data['Trata_texto'] = data['Text'].apply(lambda x: proc_texto(x))
    dados_tratados = data['Trata_texto'].values.tolist()

    # Extrair features TF-IDF com n-grams
    logger.info("Extraindo features TF-IDF...")
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
    x = tfidf.fit_transform(dados_tratados).toarray()

    # Codificar a variável de saída
    logger.info("Codificando as variáveis de saída...")
    le = LabelEncoder()
    y = le.fit_transform(data['label'])

    # Dividir dados em conjunto de treino e teste
    logger.info("Dividindo dados em treino e teste...")
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    # Inicializar e treinar modelo com hyperparameter tuning
    rf_model = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    
    logger.info("Iniciando GridSearchCV para otimização de hiperparâmetros...")
    skf = StratifiedKFold(n_splits=5)  # Aumentando o número de splits
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=skf, scoring='accuracy', n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    best_rf_model = grid_search.best_estimator_

    # Avaliar o modelo nos dados de treinamento
    logger.info("Avaliando o modelo nos dados de treinamento...")
    y_train_pred = best_rf_model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_roc_auc = roc_auc_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    train_recall = recall_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred)

    logger.info(f'Train Accuracy: {train_accuracy * 100:.2f}%')
    logger.info(f'Train ROC AUC: {train_roc_auc}')
    logger.info(f'Train F1 Score: {train_f1}')
    logger.info(f'Train Recall: {train_recall}')
    logger.info(f'Train Precision: {train_precision}')

    # Avaliar o modelo nos dados de teste
    logger.info("Avaliando o modelo nos dados de teste...")
    y_test_pred = best_rf_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_roc_auc = roc_auc_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)

    logger.info(f'Test Accuracy: {test_accuracy * 100:.2f}%')
    logger.info(f'Test ROC AUC: {test_roc_auc}')
    logger.info(f'Test F1 Score: {test_f1}')
    logger.info(f'Test Recall: {test_recall}')
    logger.info(f'Test Precision: {test_precision}')

    # Análise de erros
    logger.info("Análise de erros...")
    conf_matrix = confusion_matrix(y_test, y_test_pred)
    logger.info(f'Confusion Matrix:\n{conf_matrix}')

    # Salvar modelo e artefatos necessários
    logger.info("Salvando modelo e artefatos...")
    os.makedirs(artifacts_dir, exist_ok=True)

    model_path = os.path.join(artifacts_dir, 'model.pkl')
    tfidf_path = os.path.join(artifacts_dir, 'tfidf.pkl')
    label_encoder_path = os.path.join(artifacts_dir, 'label_encoder.pkl')

    joblib.dump(best_rf_model, model_path)
    joblib.dump(tfidf, tfidf_path)
    joblib.dump(le, label_encoder_path)

    logger.info("Modelo treinado e artefatos salvos com sucesso.")

# Chamar a função para criar e salvar o modelo
make_model()
