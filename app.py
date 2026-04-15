"""
Adaptive AI Quiz Generator - Main Flask Application
====================================================
This application generates adaptive quizzes from lecture notes using:
- Concept extraction and Knowledge Graph construction
- Multi-Armed Bandit personalization for adaptive question selection
- Multiple question formats (MCQ, True/False, Short Answer)
"""

# Suppress non-critical model loading warnings for cleaner logs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

import warnings
warnings.filterwarnings('ignore', category=UserWarning)  # Suppress model loading warnings
warnings.filterwarnings('ignore', message='.*tied weights.*')  # Suppress weight warnings
warnings.filterwarnings('ignore', message='.*UNEXPECTED.*')  # Suppress embeddings warnings

from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from functools import wraps
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
secret_key = os.getenv('SECRET_KEY')
if not secret_key:
    raise ValueError(
        'SECRET_KEY environment variable not set. '
        'Please set SECRET_KEY in your .env file for production use.'
    )
app.secret_key = secret_key

# Import blueprints
from routes.auth import auth_bp
from routes.quiz import quiz_bp
from routes.dashboard import dashboard_bp
from routes.instructor import instructor_bp
from routes.api import api_bp

# Register blueprints
app.register_blueprint(auth_bp, url_prefix='/auth')
app.register_blueprint(quiz_bp, url_prefix='/quiz')
app.register_blueprint(dashboard_bp, url_prefix='/dashboard')
app.register_blueprint(instructor_bp, url_prefix='/instructor')
app.register_blueprint(api_bp, url_prefix='/api')

# Security Headers - Protection against XSS, clickjacking, MIME-type sniffing
@app.after_request
def set_security_headers(response):
    """Add security headers to all responses"""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-inline' cdn.jsdelivr.net; style-src 'self' 'unsafe-inline' cdn.jsdelivr.net; font-src 'self' fonts.googleapis.com fonts.gstatic.com"
    return response


# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login to access this page.', 'warning')
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated_function


@app.route('/')
def index():
    """Landing page"""
    return render_template('index.html')


@app.route('/about')
def about():
    """About page explaining the system"""
    return render_template('about.html')


@app.errorhandler(404)
def page_not_found(e):
    return render_template('errors/404.html'), 404


@app.errorhandler(500)
def internal_server_error(e):
    return render_template('errors/500.html'), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port, use_reloader=False)
