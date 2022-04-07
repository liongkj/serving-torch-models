import io

import numpy as np
from tools.ocr_infer import OCRInfer, init_args, post_process

from flask import Flask, jsonify, request, send_file

app = Flask(__name__)
from base64 import encodebytes

import cv2
from PIL import Image as im


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        args = init_args()
        model = OCRInfer(det_path=args["det_path"], rec_path=args["rec_path"])
        # we will get the file from the request
        file = request.files['file'].read()
        # convert that to bytes
            # we will use the model to predict
        if file:
            example = np.fromstring(file, np.uint8)
            img = cv2.imdecode(example,cv2.IMREAD_COLOR)
            txts, boxes, debug_img = model(img)
            # convert debug img to pil image
            debug_img = post_process(debug_img)
            cv2.imwrite("debug_img.jpg", debug_img)
            pil_img = im.open("debug_img.jpg", mode='r') # reads the PIL image
            byte_arr = io.BytesIO()
            pil_img.save(byte_arr, format='PNG') 
            encoded = encodebytes(byte_arr.getvalue()).decode('ascii')
            
            return {'preds': txts, 'img': encoded}
