from flask import Flask, request, render_template, url_for, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2



app = Flask(__name__)



classes = ['malignant' ,'benine']
model=load_model('F:/intel-Classfier-main/fatma123.h5')

@app.route('/')
def index():

    return render_template('index.html', appName="fatma123")


@app.route('/predictApi', methods=["POST"])
def api():
    # Get the image from post request

        if 'fileup' not in request.files:
            return "Please try again. The Image doesn't exist"
        image = request.files.get('fileup')


        img = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (70, 70))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        print("Model predicting ...")
        result = model.predict(img)
        print("Model predicted")
        ind = np.argmax(result)
        prediction = classes[ind]
        print(result)
        print(prediction)
        return jsonify({'prediction': prediction})



@app.route('/predict', methods=['GET', 'POST'])
def predict():
    print("run code")
    if request.method == 'POST':
        # Get the image from post request
        print("image loading....")
        image = request.files['fileup']
        print("image loaded....")
        img = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (70, 70))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        print("predicting ...")
        result = model.predict(img)
        print("predicted ...")
        ind = np.argmax(result)
        prediction = classes[ind]

        print(prediction)

        return render_template('index.html', prediction=prediction, image='static/IMG/', appName="fatma123")
    else:
        return render_template('index.html',appName="fatma123.")


if __name__ == '__main__':
    app.run(debug=True)