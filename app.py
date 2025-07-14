from flask import Flask
from routes.video_routes import video_bp
import os
import tempfile

app = Flask(__name__)
TEMP_DIR = tempfile.gettempdir()
os.makedirs('static', exist_ok=True)

app.register_blueprint(video_bp)

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
