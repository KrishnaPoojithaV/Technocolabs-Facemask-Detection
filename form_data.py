import json,time

from keras.models import load_model

from camera import Camera
from flask import Flask, render_template, request, jsonify, Response
import requests
import base64,cv2

app = Flask(__name__)

MODEL_PATH = 'model-016.model'
model = load_model(MODEL_PATH)
model._make_predict_function()
print('Model loaded. Check http://127.0.0.1:5000/')


camera = None

def get_camera():
    global camera
    if not camera:
        camera = Camera()

    return camera


output = []
@app.route('/')
def home_page():
    return render_template('base.html')

def gen(camera):
    while True:
        data = camera.get_feed()

        frame=data
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video/')
def video():
    camera = get_camera()
    return Response(gen(camera), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=False, threaded = False)
