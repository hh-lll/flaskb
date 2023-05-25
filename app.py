from flask import Flask, render_template, request, Response, jsonify
from werkzeug.utils import secure_filename
import algorithm.openCV1 as test
import os
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import ForeignKey

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'upload/'
app.config[
    'SQLALCHEMY_DATABASE_URI'] = "mysql+pymysql://root:20220325Tx@gz-cdb-hay8j0mh.sql.tencentcdb.com:63831/bracket"
db = SQLAlchemy(app)

# if __name__ == '__main__':

@app.route('/')
def upload_file():
    # test.min("upload/", f.filename)
    return render_template('upload.html')


@app.route('/hello')
def hello():
    return "你好"


@app.route('/success')
def success_api(msg: str = "成功"):
    """ 成功响应 默认值”成功“ """
    return jsonify(success=True, msg=msg)


@app.route('/download')
def download():
    pic = request.args['pic']
    if pic.startswith("DstLIne_"):
        picExists = os.path.exists("images/Dst/" + pic)
    elif pic.startswith("Base_"):
        picExists = os.path.exists("images/Base/" + pic)
    else:
        picExists = os.path.exists("upload/" + pic)
    if not picExists:
        return "can't find picture"
    if pic.startswith("DstLIne_"):
        image = open("images/Dst/" + pic, 'rb')
    elif pic.startswith("Base_"):
        image = open("images/Base/" + pic, 'rb')
    else:
        image = open("upload/" + pic, 'rb')
    # image2 = open("images/Dst/"+"DstLIne_"+f.filename,'rb')
    # print(image)
    # image = file("../images/Base/"+"Base"+f.filename)
    resp = Response(image, mimetype="image/jpeg")
    # return "hh"
    return resp


@app.route('/uploader', methods=['GET', 'POST'])
def uploader():
    if request.method == 'POST':
        f = request.files['file']
        print(request.files)
        f.save(os.path.join("upload/", secure_filename(f.filename)))
        print("/upload/" + f.filename)
        data = test.min("upload/", f.filename)
        return data
    else:
        return render_template('upload.html')

class User(db.Model):
    # 定义表名
    __tablename__ = 'user'
    # 定义字段
    user_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(64))
    password = db.Column(db.String(64))
    type = db.Column(db.String(64))


class Evaluation(db.Model):
    # 定义表名
    __tablename__ = 'evaluation'
    # 定义字段
    evaluation_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    cut_file = db.Column(db.String(64))
    final_file = db.Column(db.String(64))
    total_score = db.Column(db.Double)
    date = db.Column(db.String(64))
    time = db.Column(db.String(64))
    user_id = db.Column(db.Integer, ForeignKey("user.user_id"))


class Offset(db.Model):
    # 定义表名
    __tablename__ = 'offset'
    # 定义字段
    offset_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    horizontal = db.Column(db.Integer)
    shaftAngle = db.Column(db.Integer)
    vertical = db.Column(db.Integer)
    type = db.Column(db.String(64))
    evaluation_id = db.Column(db.Integer, ForeignKey("evaluation.evaluation_id"))

class Score(db.Model):
    # 定义表名
    __tablename__ = 'score'
    # 定义字段
    score_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    horizontal = db.Column(db.Integer)
    shaftAngle = db.Column(db.Integer)
    vertical = db.Column(db.Integer)
    score = db.Column(db.Integer)
    type = db.Column(db.String(64))
    evaluation_id = db.Column(db.Integer, ForeignKey("evaluation.evaluation_id"))

@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


@app.route('/logup', methods=['POST'])
def logup():
    user_name = request.form.get('username')
    user_password = request.form.get('password')
    # user_phone = request.form.get('type')
    user = User()
    user.name = user_name
    user.password = user_password
    user.type = "1"
    db.session.add(user)
    db.session.commit()
    return "注册成功"

@app.route('/login', methods=['POST'])
def logup():
    user_name = request.form.get('username')
    user_password = request.form.get('password')
    # user_phone = request.form.get('type')

    user = db.session.query(User).filter_by(name=user_name, password = user_password).all()
    if user == None:
        return "登录失败"
    else:
        return "登录成功"

if __name__ == '__main__':
    app.run()
