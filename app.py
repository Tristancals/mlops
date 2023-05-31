from io import BytesIO
from sklearn.datasets import load_digits
import os, shutil
import pickle
import calendar
import time
import numpy as np
from flask import Flask, render_template, request
from PIL import Image
from PIL.ImageOps import invert
import base64

from prometheus_client import Summary, start_http_server, Counter

app = Flask(__name__)

ZZZ_TEST_PROCESS_IMAGE = Summary("ZZZ_TEST_PROCESS_IMAGE", "time for process image")
COUNTERS = [Counter(f'Predicted_{i}_count', f'Number of {i} predicted') for i in range(10)]



def create_new_model():
    digits = load_digits()
    print(digits.data.shape)

def remove_temp_img():
    print("remove_temp_img")
    folder = './temp_data'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


@app.get("/")
def display_index():
    print("display_index")
    return render_template('index.html')


@ZZZ_TEST_PROCESS_IMAGE.time()
def process_image(data):
    print("process_image")

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

    return img


def find_digit(lettre):
    print("find_digit")
    charstr = '0123456789abcdefghijklmnopqrstuvwxyz'
    return charstr.index(lettre.lower())


def find_lettre(digit):
    print("find_lettre")
    charstr = '0123456789abcdefghijklmnopqrstuvwxyz'
    return charstr[digit]


@app.post("/api/fix")
def add_data_api():
    print("add_data_api")
    data = request.get_json()
    print(data)
    print(len(data['value']))
    if len(data["value"]) == 1 :
        print("lkljj")
    target = find_digit(data['value'])
    src_dir = "./temp_data/"+data["temp_img"]+".png"
    dst_dir = "./new_data/"+data["temp_img"]+"_"+str(target)+".png"
    shutil.copy(src_dir, dst_dir)

    return {"value":"proute"}

@app.post("/api/predict")
def predict_api():
    print("predict_api")

    data = request.get_json()
    data = str(data)
    data = data.split(",")
    data = data[1]
    data = data.replace("\'}", "")
    img = process_image(data)

    na = np.array(img)
    na = np.insert(na, 0, 0, axis=1)
    na = np.insert(na, 7, 0, axis=1)
    img = Image.fromarray(na)
    img.save("images/rowadd.png")

    img_number = na.reshape(-1)
    print(len(img_number))

    img_number = (img_number / float(max(img_number)) * 15).round()
    print(img_number)

    loaded_model = pickle.load(open('digit_classifier.pkl', 'rb'))

    result = loaded_model.predict([img_number])
    remove_temp_img()
    time_stamp = calendar.timegm(time.gmtime())
    img.save("temp_data/" + str(time_stamp) + ".png")
    print(result)
    COUNTERS[int(result[0])].inc()

    return {'result': str(find_lettre(result[0])), 'time_stamp': time_stamp}


if __name__ == '__main__':
    start_http_server(8000)
    app.run(None, 3000)
