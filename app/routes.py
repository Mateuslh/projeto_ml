from flask import Blueprint, request, jsonify

from modelo.func import predict_text

main_bp = Blueprint('main', __name__)


@main_bp.route('/', methods=['POST'])
def is_fake():
    try:
        text = request.get_json()

        if text is None or text == '':
            return jsonify({'success': False, 'result': 'No texto passed'}), 406
    except Exception as e:
        return jsonify({'success': False, 'result': str(e)}), 400

    try:
        result = predict_text(text)
    except Exception as e:
        return jsonify({'success': False, 'result': str(e)}), 422

    return jsonify({'success': False, 'result': result}), 200
