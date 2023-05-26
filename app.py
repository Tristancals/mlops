from io import BytesIO
import pickle

import numpy as np
from flask import Flask, render_template, jsonify, request, url_for
from PIL import Image
from PIL.ImageOps import invert
import matplotlib.pyplot as plt
import base64

app = Flask(__name__)


@app.get("/")
def display_index():
    return render_template('index.html',result_predict = "rien pour le moment")


@app.post("/api/predict")
def predict_api():
    data = request.get_json()
    data = str(data)
    data = data.split(",")
    data = data[1]
    data = data.replace("\'}", "")

    img = base64.b64decode(data)
    img = Image.open(BytesIO(img))
    img.save("images/original.png")

    img = img.convert(mode="L")
    img.save("images/convert.png")

    img = invert(img)
    img.save("images/revert.png")

    bbox = img.getbbox()
    img = img.crop(bbox)
    img.save("images/getbox.png")

    newsize = (6, 8)
    img = img.resize(newsize)
    img.save("images/resize.png")

    na = np.array(img)
    na = np.insert(na, 0, 0, axis=1)
    na = np.insert(na, 7, 0, axis=1)
    img = Image.fromarray(na)
    img.save("images/rowadd.png")

    img_number = na.reshape(-1)
    print(len(img_number))
    print(img_number)

    loaded_model = pickle.load(open('digit_classifier.pkl','rb'))

    result = loaded_model.predict([img_number])

    print(result)

    return {'result': str(result[0])}

