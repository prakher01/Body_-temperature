from flask import Flask, render_template
from routes.video_routes import video_bp

app = Flask(__name__)
app.register_blueprint(video_bp)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, threaded=True, port=5001)
