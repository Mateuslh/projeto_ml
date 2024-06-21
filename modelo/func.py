import os
import time

import joblib


current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, os.pardir, 'message_classifier.pkl')

model_data = joblib.load(file_path)
model = model_data['model']
metrics = model_data['metrics']


def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        exec_time = f"{int((time.time() - start_time) * 1000)}ms"
        result['execution_time'] = exec_time
        return result

    return wrapper


@measure_time
def predict_text(text):
    prediction = float(model.predict([text])[0])
    probability = model.predict_proba([text])[0].max()

    confidence_interval = [probability - 0.05, probability + 0.05]

    result = {
        'input_message': text,
        'prediction': prediction,
        'model': {
            'probability': probability,
            'confidence_interval': confidence_interval,
            'accuracy': metrics['accuracy'],
            'auc': metrics['auc'],
            'f1_score': metrics['f1_score'],
            'recall': metrics['recall'],
            'precision': metrics['precision'],
            'feature_importance': metrics['feature_importance'],
            'info': {
                'model_name': 'MessageClassifier',
                'model_version': '1.0'
            }
        }
    }
    return result
