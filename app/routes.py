from flask import Blueprint, request, jsonify
import modelo.app as predict  # Supondo que 'predict' seja o nome do arquivo onde está definida a função de previsão
from flask_cors import CORS

main_bp = Blueprint('main', __name__)
CORS(main_bp)

@main_bp.route('/predict', methods=['POST'])
def is_fake():
    try:
        data = request.get_json()
        if 'noticia' not in data:
            return jsonify({'success': False, 'result': 'No "noticia" field provided in the request.'}), 400

        noticia = data['noticia']
        resultado = predict.predict(noticia)

        return jsonify({'success': True, 'resultado': resultado}), 200

    except Exception as e:
        return jsonify({'success': False, 'result': str(e)}), 500
