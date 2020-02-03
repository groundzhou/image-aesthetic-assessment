from app.database import db


class User(db.Model):
    __tablename__ = 'user'

    openid = db.Column(db.Integer, primary_key=True)
