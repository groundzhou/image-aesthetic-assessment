from flask import Blueprint, jsonify, request
import json

bp = Blueprint('api', __name__)


# 评分
@bp.route('/', methods=['GET', 'POST'])
def score():
    if request.method == 'POST':
        openid = None
        image = None
        # 从表单获取
        if request.form:
            openid = request.form.get('openid')
            image = request.form.get('image')
        # 从数据段获取
        if request.data:
            data = json.loads(request.data)
            openid = data.get('openid')
            image = data.get('image')

        if not openid:
            return jsonify(code=1,
                           status=400,
                           message='User openid is required.'), 400
        if not image:
            return jsonify(code=2,
                           status=400,
                           message='An image is required.'), 400

        return jsonify(code=0,
                       message='success',
                       data={'score': 7,
                             'light': 9.2,
                             'color': 6})

    if request.method == 'GET':
        return '<h1>Image Aesthetic Assessment System API</h1>'


# 注册时记录用户openid
@bp.route('/register/<int:openid>', methods=['POST'])
def register(openid):
    return jsonify(code=0,
                   message='success')