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
            data['remark'] = '敏感的色彩感知，曝光合适，色彩搭配愉悦人心。'
        elif result['aesthetic']['score'] > 5:
            data['remark'] = '作品一般，拍摄角度好。'
        elif result['aesthetic']['score'] > 4.5:
            data['remark'] = '作品一般。'
        else:
            data['remark'] = '作品较差，摄影技术有待提高。'

        if result['aesthetic']['std'] > 1.4:
            data['remark'] = data['remark'] + '审美独特，对于拍照，有自己深入的思考，拍摄作品容易出彩。'
        else:
            data['remark'] = data['remark'] + '审美大众化，拍摄作品平中出奇，迎合大众的审美情趣。'

        tags = [k for k, v in result['style'].items() if v > 0.1]
        tag_cn = {'Complementary_Colors': '互补色',
                  'Duotones': '双色调',
                  'HDR': 'HDR',
                  'Image_Grain': '图像纹理',
                  'Light_On_White': '白光补色',
                  'Long_Exposure': '长曝光',
                  'Macro': '微距',
                  'Motion_Blur': '动态模糊',
                  'Negative_Image': '负片',
                  'Rule_of_Thirds': '三分法',
                  'Shallow_DOF': '浅景深',
                  'Silhouettes': '剪影',
                  'Soft_Focus': '柔焦',
                  'Vanishing_Point': '消失点'}
        tag_remarks = {'Complementary_Colors': '互补色搭配，引起强烈色差，色彩更具美感。',
                       'Duotones': '善用双色调，颜色层次特殊，照片更有趣。',
                       'HDR': '高动态范围图像，利用每个曝光细节，反应真实环境。',
                       'Image_Grain': '借助图像纹理，添加特效，增强照片质感。',
                       'Light_On_White': '白光补色，尤生温暖、爽快之感。',
                       'Long_Exposure': '长曝光拍摄，景象更清晰，照片更具梦幻感。',
                       'Macro': '灵活运用微距，将摄影细节展现得淋漓尽致。',
                       'Motion_Blur': '运用动态模糊，抓拍拖动痕迹，表现速度感。',
                       'Negative_Image': '利用负片，生成反色照片，拍出不一样的色彩。',
                       'Rule_of_Thirds': '符合构图三分法。',
                       'Shallow_DOF': '浅景深使照片更加纯粹。',
                       'Silhouettes': '运用剪影，突出景物轮廓，增加真实感。',
                       'Soft_Focus': '通过柔焦效果，掩盖景物缺陷，增加照片柔美气氛和艺术效果。',
                       'Vanishing_Point': '消失点摄影，通过远景增加照片的立体感。'}
        data['tags'] = [tag_cn[i] for i in tags]
        data['remark'] = data['remark'] + ''.join([tag_remarks[i] for i in tags])

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
