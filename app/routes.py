from flask import Blueprint, request, jsonify
from flask_cors import CORS
import modelo.app as app  # Consome o primeiro código
import modelo.func as func  # Consome o segundo código

main_bp = Blueprint('main', __name__)
CORS(main_bp)

@main_bp.route('/predict_floresta', methods=['POST'])
def predict1():
    try:
        data = request.get_json()
        if 'noticia' not in data:
            return jsonify({'success': False, 'result': 'No "noticia" field provided in the request.'}), 400

        noticia = data['noticia']
        resultado = func.predict_label(noticia, func.model, func.tfidf, func.label_encoder)

        return jsonify({'success': True, 'resultado': resultado}), 200

    except Exception as e:
        return jsonify({'success': False, 'result': str(e)}), 500

@main_bp.route('/predict_regressao', methods=['POST'])
def predict2():
    try:
        data = request.get_json()
        if 'noticia' not in data:
            return jsonify({'success': False, 'result': 'No "noticia" field provided in the request.'}), 400

        noticia = data['noticia']
        resultado = app.predict(noticia)

        return jsonify({'success': True, 'resultado': resultado}), 200

    except Exception as e:
        return jsonify({'success': False, 'result': str(e)}), 500
