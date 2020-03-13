import json
from flask import Blueprint, jsonify, request
from PIL import Image

from src.evaluator import InferenceModel

bp = Blueprint('api', __name__)
model = InferenceModel(aesthetic_model_path='./checkpoints/aesthetic-epoch-44.pth',
                       style_model_path='./checkpoints/style-epoch-45.pth')


# 评分
@bp.route('/', methods=['POST'])
def score():
    if request.method == 'POST':
        openid = None
        image_upload = None
        if request.form:
            # 获取openid
            openid = request.form.get('openid')
            # 获取图片
            image_upload = request.files.get('image')

        if not openid:
            return jsonify(code=1,
                           status=400,
                           message='User openid is required.'), 400
        if not image_upload or not allowed_file(image_upload.filename):
            return jsonify(code=2,
                           status=400,
                           message='An image(jpg, png) is required.'), 400

        image = Image.open(image_upload).convert('RGB')
        result = model.predict(image)
        data = {'aesthetic': result['aesthetic']}
        if result['aesthetic']['score'] > 6:
            data['remark'] = '高超的构图技巧，精湛的细节处理，精彩瞬间，一拍即触。'
        elif result['aesthetic']['score'] > 5.5:
            data['remark'] = '作品比较优秀。'
        else:
            data['remark'] = '作品一般'
        if result['aesthetic']['std'] > 1.4:
            data['remark'] = data['remark'] + '审美独特，对于拍照，有自己深入的思考，拍摄作品容易出彩。'
        else:
            data['remark'] = data['remark'] + '审美大众化，拍摄作品平中出奇，迎合大众的审美情趣。'
        data['tags'] = [' '.join(k.split('_')) for k, v in result['style'].items() if v > 0.1]

        return jsonify(code=0,
                       message='success',
                       data=data)


# 注册时记录用户openid
@bp.route('/register/<int:openid>', methods=['POST'])
def register(openid):
    return jsonify(code=0,
                   message='success')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['png', 'jpg', 'jpeg']