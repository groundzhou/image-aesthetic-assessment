import os
from flask import Flask, request


def create_app(test_config=None):
    # create and config the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        SQLALCHEMY_DATABASE_URI='sqlite:///'+os.path.join(app.instance_path, 'aesthetics.db'),
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
    )

    if not test_config:
        # load the instance config, if exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # initialize app with database
    from webapp.database import init_app
    init_app(app)

    # a simple page that says hello
    @app.route('/hello')
    def hello():
        return 'Hello, World!'

    # register blueprints
    from webapp import api
    app.register_blueprint(api.bp)

    # 解决跨域问题
    app.after_request(cors)

    return app


def cors(res):
    res.headers['Access-Control-Allow-Origin'] = request.headers.get('Origin', default='*')
    res.headers["Access-Control-Allow-Credentials"] = 'true'
    return res
