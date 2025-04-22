# auth_service/app.py
from flask import Flask, request, jsonify
import jwt
import datetime
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URI', 'sqlite:///users.db')
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True)
    password = db.Column(db.String(100))

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    user = User.query.filter_by(username=data['username']).first()
    
    if user and user.password == data['password']:  # Use proper password hashing in production
        token = jwt.encode({
            'user_id': user.id,
            'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
        }, app.config['SECRET_KEY'])
        
        return jsonify({'token': token})
    
    return jsonify({'message': 'Invalid credentials'}), 401

@app.route('/verify', methods=['POST'])
def verify():
    data = request.get_json()
    try:
        payload = jwt.decode(data['token'], app.config['SECRET_KEY'], algorithms=['HS256'])
        user = User.query.filter_by(id=payload['user_id']).first()
        return jsonify({'user_id': user.id, 'username': user.username})
    except:
        return jsonify({'message': 'Invalid token'}), 401

if __name__ == '__main__':
    db.create_all()
    app.run(host='0.0.0.0', port=5000)