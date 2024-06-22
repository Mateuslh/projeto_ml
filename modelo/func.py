import time
import os
import joblib
import re
import unicodedata
from nltk.corpus import stopwords

# Determine o diretório base do projeto
BASE_DIR = os.getcwd()

# Construa o caminho completo para os arquivos .pkl e outros artefatos
tfidf_path = os.path.join(BASE_DIR, 'artifacts', 'tfidf.pkl')
model_path = os.path.join(BASE_DIR, 'artifacts', 'model.pkl')
label_encoder_path = os.path.join(BASE_DIR, 'artifacts', 'label_encoder.pkl')

# Carregue o modelo, vetorizador TF-IDF e LabelEncoder
try:
    model = joblib.load(model_path)
    tfidf = joblib.load(tfidf_path)
    label_encoder = joblib.load(label_encoder_path)
except Exception as e:
    print(f"Erro ao carregar modelo e artefatos: {e}")
    model = None
    tfidf = None
    label_encoder = None

def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        exec_time = f"{int((time.time() - start_time) * 1000)}ms"
        result['execution_time'] = exec_time
        return result
    return wrapper

def proc_texto(texto):
    texto = unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('ascii')
    texto = texto.lower()
    texto = re.sub(r'[^\w\s]', '', texto)
    texto = texto.split()

    stplist = stopwords.words('english')
    stplist = [word.encode('ascii', 'ignore').decode('ascii') for word in stplist]
    stplist = [word.lower() for word in stplist]

    texto = [palavra for palavra in texto if palavra not in stplist]
    # Remover palavras muito curtas e muito longas
    texto = [palavra for palavra in texto if len(palavra) > 1 and len(palavra) < 20]

    return ' '.join(texto)

@measure_time
def predict_text(text):
    if model is None or tfidf is None or label_encoder is None:
        return {
            'error': 'Modelo não foi carregado corretamente. Verifique os arquivos .pkl e artefatos.'
        }

    try:
        # Pré-processamento do texto de entrada
        processed_text = proc_texto(text)

        # Vetorização do texto usando TF-IDF
        text_vectorized = tfidf.transform([processed_text]).toarray()

        # Predição usando o modelo carregado
        prediction = model.predict(text_vectorized)[0]
        probability = model.predict_proba(text_vectorized)[0].max()

        confidence_interval = [probability - 0.05, probability + 0.05]

        # Decodificação da predição usando LabelEncoder
        prediction_label = label_encoder.inverse_transform([prediction])[0]

        result = {
            'input_message': text,
            'prediction': prediction_label,
            'model': {
                'probability': probability,
                'confidence_interval': confidence_interval,
                'info': {
                    'model_name': 'MessageClassifier',
                    'model_version': '1.0'
                }
            }
        }
        return result

    except Exception as e:
        return {
            'error': f"Erro durante a previsão: {e}"
        }
