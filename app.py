from flask import Flask,render_template,request,jsonify
import json
import base64
import cv2
import numpy as np
from prediction import Prediction

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload")
def upload():
    return render_template("predict.html")

@app.route('/predict',methods=['POST'])
def predict():
    with open('predictions.json') as f:
        prediction = json.load(f)
    image = request.json['image']
    img = base64.b64decode(image)
    jpg_as_np = np.frombuffer(img, dtype=np.uint8)
    img = cv2.imdecode(jpg_as_np, flags=1)
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    if len(eye_cascade.detectMultiScale(img)) != 0:
        ex,ey,ew,eh = eye_cascade.detectMultiScale(img)[0]
        img_2 = img[ey:ey+eh,ex:ex+ew]
        img_2 = cv2.resize(img_2,(224,224))
        pred = Prediction().classify(img_2)
        return jsonify([{'image':prediction[str(pred)]}])
    else:
        syntax =[{
            'image' : "<h6 class='text-center py-4' style='color: red;'>Eye is not recognized properly, retake and upload again!</h6>"
        }]
        
        return jsonify(syntax)

if __name__ == '__main__':
    app.run(debug=True)