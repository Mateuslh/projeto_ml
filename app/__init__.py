from flask import Flask


def create_app():
    app = Flask(__name__)

    # Configurações podem ser carregadas aqui
    app.config.from_object('app.config.Config')

    # Registra os blueprints
    from app.routes import main_bp
    app.register_blueprint(main_bp)

    return app
