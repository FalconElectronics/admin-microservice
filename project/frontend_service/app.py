# frontend_service/app.py
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_login import LoginManager, current_user, login_user, logout_user, login_required
import requests
import os
import json
import redis

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key')

# Redis client for inter-service communication
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=0
)

# Service URLs
AUTH_SERVICE_URL = os.getenv("AUTH_SERVICE_URL", "http://auth:5000")
WEAPON_SERVICE_URL = os.getenv("WEAPON_SERVICE_URL", "http://weapon:5001")
CROWD_SERVICE_URL = os.getenv("CROWD_SERVICE_URL", "http://crowd:5002")
SPEECH_SERVICE_URL = os.getenv("SPEECH_SERVICE_URL", "http://speech:5003")

# Setup Login Manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User:
    def __init__(self, id, username):
        self.id = id
        self.username = username
        self.is_authenticated = True
        self.is_active = True
        self.is_anonymous = False
    
    def get_id(self):
        return str(self.id)

@login_manager.user_loader
def load_user(user_id):
    # Fetch user data from auth service
    response = requests.get(f"{AUTH_SERVICE_URL}/user/{user_id}")
    if response.status_code == 200:
        data = response.json()
        return User(data['id'], data['username'])
    return None

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        response = requests.post(f"{AUTH_SERVICE_URL}/login", json={
            'username': username,
            'password': password
        })
        
        if response.status_code == 200:
            data = response.json()
            # Verify token and get user info
            user_response = requests.post(f"{AUTH_SERVICE_URL}/verify", json={
                'token': data['token']
            })
            
            if user_response.status_code == 200:
                user_data = user_response.json()
                user = User(user_data['user_id'], user_data['username'])
                login_user(user)
                return redirect(url_for('dashboard'))
        
        return render_template('accounts/login.html', msg='Invalid username or password')
    
    return render_template('accounts/login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('home/index.html', segment='index')

@app.route('/weapon-detection')
@login_required
def weapon_detection():
    return render_template('home/weapon-detection.html', segment='weapon')

@app.route('/crowd-analytics')
@login_required
def crowd_analytics():
    return render_template('home/crowd-analytics.html', segment='crowd')

@app.route('/speech-analytics')
@login_required
def speech_analytics():
    return render_template('home/env-analytics.html', segment='env')

@app.route('/action-analytics')
@login_required
def action_analytics():
    return render_template('home/action-analytics.html', segment='action')

# API routes to proxy requests to microservices
@app.route('/api/weapon-detection')
@login_required
def api_weapon_detection():
    response = requests.get(f"{WEAPON_SERVICE_URL}/detection")
    return jsonify(response.json())

@app.route('/api/crowd-analytics')
@login_required
def api_crowd_analytics():
    response = requests.get(f"{CROWD_SERVICE_URL}/analytics")
    return jsonify(response.json())

@app.route('/api/speech-analytics')
@login_required
def api_speech_analytics():
    response = requests.get(f"{SPEECH_SERVICE_URL}/speech")
    return jsonify(response.json())

@app.route('/api/action-analytics')
@login_required
def api_action_analytics():
    response = requests.get(f"{SPEECH_SERVICE_URL}/action")
    return jsonify(response.json())

@app.route('/video_feed/<cam_id>')
@login_required
def video_feed(cam_id):
    # Redirect to the appropriate service based on the camera feed type
    service_url = CROWD_SERVICE_URL
    return redirect(f"{service_url}/frame/{cam_id}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)