from flask import Blueprint, request, Response, jsonify
import os
import tempfile
import threading
import queue
import json
from services.video_processing import (
    set_kalman_defaults,
    broadcast_rgb_data,
    generate_gray_frames,
    rgb_clients,
    rgb_data_lock,
    current_rgb_data
)

video_bp = Blueprint('video', __name__)
video_path = ""

@video_bp.route('/set_variable', methods=['POST'])
def set_variable():
    data = request.get_json()
    variable = data.get('variable', None)
    print(f"Variable received: {variable}")
    set_kalman_defaults()
    return jsonify({'status': 'Variable received successfully'})

@video_bp.route('/upload', methods=['POST'])
def upload():
    global video_path
    video_file = request.files['video']
    temp_dir = tempfile.gettempdir()
    video_path = os.path.join(temp_dir, video_file.filename)
    video_file.save(video_path)
    return "success"

@video_bp.route('/video_feed')
def video_feed():
    global video_path
    if video_path:
        return Response(generate_gray_frames(video_path),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    return "No video", 404

@video_bp.route('/rgb_stream')
def rgb_stream():
    def event_stream():
        client_queue = queue.Queue(maxsize=10)
        rgb_clients.append(client_queue)
        try:
            while True:
                try:
                    data = client_queue.get(timeout=30)
                    yield f"data: {json.dumps(data)}\n\n"
                except queue.Empty:
                    yield "data: {\"heartbeat\": true}\n\n"
        finally:
            if client_queue in rgb_clients:
                rgb_clients.remove(client_queue)
    return Response(event_stream(), mimetype='text/event-stream')

@video_bp.route('/get_rgb_data')
def get_rgb_data():
    with rgb_data_lock:
        return jsonify(current_rgb_data)
