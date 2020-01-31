from flask import Blueprint, jsonify

bp = Blueprint('api', __name__)


@bp.route('/', methods=['GET'])
def score():
    return jsonify(code=0,
                   message='api test',
                   data={'score': 7,
                         'light': 9.2,
                         'color': 6})
