import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


def make_model():
    messages = [
        "This is a real message.",
        "Win a million dollars now!",
        "New species discovered.",
        "Flat earth theory is true.",
        "Breaking news: stock prices soar.",
        "Exclusive offer just for you.",
        "Scientists find cure for disease.",
        "Click here to claim your prize.",
        "Government announces new policy.",
        "Learn how to make money online.",
        "Celebrity spotted at local restaurant.",
        "New technology revolutionizes industry."
    ]
    labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1]

    # Divide os dados em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(messages, labels, test_size=0.2, random_state=42)

    # Cria um pipeline de processamento de texto e modelo, sem remover stop words
    pipeline = make_pipeline(TfidfVectorizer(stop_words=None), RandomForestClassifier())

    # Treina o modelo
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    feature_importance = {
        'length': 1,  # preencher aqui
        'keywords': 1,  # preencher aqui
        'punctuation': 1,  # preencher aqui
    }

    total_importance = sum(feature_importance.values())
    feature_importance = {k: v / total_importance for k, v in feature_importance.items()}

    joblib.dump({
        'model': pipeline,
        'metrics': {
            'accuracy': accuracy,
            'auc': auc,
            'f1_score': f1,
            'recall': recall,
            'precision': precision,
            'feature_importance': feature_importance
        }
    }, 'message_classifier.pkl')
