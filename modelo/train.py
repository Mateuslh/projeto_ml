import os
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import unicodedata
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
import logging

# Configuração do logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Baixar recursos do NLTK necessários
nltk.download('stopwords')
nltk.download('wordnet')

# Função para pré-processamento do texto
def preprocess_text(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    text = re.sub(r'http\S+', '', text)  # Remover URLs
    text = re.sub(r'@\w+', '', text)     # Remover menções
    text = re.sub(r'\d+', '', text)      # Remover números
    text = re.sub(r'[^\w\s]', '', text)  # Remover caracteres não-alfabéticos
    text = text.lower()                  # Converter para minúsculas

    # Tokenização e lematização
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
    
    return ' '.join(words)

# Função para treinar o modelo
def train_model():
    logger.info("Iniciando a criação do modelo...")

    # Diretório de artefatos
    artifacts_dir = os.path.join(os.getcwd(), 'artifacts')
    os.makedirs(artifacts_dir, exist_ok=True)

    # Caminho para os arquivos de artefatos
    model_path = os.path.join(artifacts_dir, 'model.pkl')
    tfidf_path = os.path.join(artifacts_dir, 'tfidf.pkl')
    label_encoder_path = os.path.join(artifacts_dir, 'label_encoder.pkl')

    # Carregar os dados
    data_path = os.path.join(os.getcwd(), 'data', 'fake_and_real_news.csv')
    data = pd.read_csv(data_path)

    # Usar apenas uma fração dos dados (opcional)
    data = data.sample(frac=0.3, random_state=42)

    logger.info(f"Dados carregados com sucesso. {data.shape[0]} linhas e {data.shape[1]} colunas.")

    # Remover duplicatas
    logger.info(f"Removendo duplicatas...")
    data.drop_duplicates(inplace=True)
    logger.info(f"Linhas duplicadas removidas. Novo tamanho do conjunto de dados: {data.shape}")

    # Pré-processar os textos
    logger.info("Pré-processando os textos...")
    data['processed_text'] = data['Text'].apply(preprocess_text)

    # Preparar dados para treino
    X = data['processed_text'].values
    y = data['label'].values

    # Label Encoding dos rótulos
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # TF-IDF Vectorizer
    logger.info("Criando TF-IDF Vectorizer...")
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = tfidf_vectorizer.fit_transform(X)

    # Dividir dados em conjunto de treino e teste
    logger.info("Dividindo dados em conjunto de treino e teste...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Inicializar modelo RandomForest
    rf_model = RandomForestClassifier(random_state=42)

    # Definir grid de hiperparâmetros para GridSearchCV
    param_grid = {
        'n_estimators': [100],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    
    logger.info("Iniciando GridSearchCV para otimização de hiperparâmetros...")
    skf = StratifiedKFold(n_splits=5)  # Aumentando o número de splits
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=skf, scoring='accuracy', n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    best_rf_model = grid_search.best_estimator_

    # Configurar GridSearchCV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=skf, scoring='accuracy', n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    # Melhor modelo encontrado pelo GridSearchCV
    best_rf_model = grid_search.best_estimator_

    # Avaliar o modelo nos dados de treino
    logger.info("Avaliando o modelo nos dados de treino...")
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

    # Matriz de confusão
    logger.info("Matriz de Confusão:")
    conf_matrix = confusion_matrix(y_test, y_test_pred)
    logger.info(conf_matrix)

    # Salvar modelo e artefatos necessários
    logger.info("Salvando modelo e artefatos...")
    joblib.dump(best_rf_model, model_path)
    joblib.dump(tfidf_vectorizer, tfidf_path)
    joblib.dump(label_encoder, label_encoder_path)

    logger.info("Modelo treinado e artefatos salvos com sucesso.")

# Chamar a função para treinar o modelo
if __name__ == "__main__":
    train_model()
