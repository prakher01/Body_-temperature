from flask import Blueprint, render_template, request, Response, send_from_directory
from utils.processing import process_video
import uuid
import os
import tempfile

video_bp = Blueprint('video', __name__)
TEMP_DIR = tempfile.gettempdir()

@video_bp.route('/')
def index():
    return render_template('index.html')

@video_bp.route('/upload', methods=['POST'])
def upload():
    video_file = request.files['video']
    temp_filename = f"{uuid.uuid4()}.mp4"
    temp_path = os.path.join(TEMP_DIR, temp_filename)
    video_file.save(temp_path)
    return Response(process_video(temp_path), mimetype='text/event-stream')

@video_bp.route('/static/<filename>')
def serve_static(filename):
    return send_from_directory('static', filename)
