# -*- coding: utf-8 -*-

import flask
from flask import Flask, request
from keras.models import load_model
import numpy as np
import cv2

api = Flask(__name__)

# 4-predict
def model_predict_type_one(imagePath):
    temp = cv2.imread(imagePath, cv2.IMREAD_COLOR)
    temp =np.array(temp)
    temp = temp.astype('float32')
    temp =temp/ 255
    temp = cv2.resize(temp, (128, 128), interpolation = cv2.INTER_AREA)
    temp=np.array(temp)
    temp = temp.reshape(1, 128, 128, 3)    
    model = load_model('/models/weights25.h5')
    return model.predict(temp).argmax(axis=-1)[0]

# 2-predict + 3-predict
def model_predict_type_two(imagePath):
    predict=-1
    temp = cv2.imread(imagePath, cv2.IMREAD_COLOR)
    temp =np.array(temp)
    temp = temp.astype('float32')
    temp =temp/ 255
    temp = cv2.resize(temp, (128, 128), interpolation = cv2.INTER_AREA)
    temp=np.array(temp)
    temp = temp.reshape(1, 128, 128, 3)    
    model = load_model('/models/weights20.h5')
    model1 = load_model('/models/weights30.h5')
    ctype= model.predict(temp).argmax(axis=-1)[0]
    if ctype==0:
      predict=0
    if ctype==1:
        predict=model1.predict(temp).argmax(axis=-1)[0]+1
    return predict

@api.route('/prediction', methods=["POST"])
def classify_grade():    
    data = {"success": False, "predict": -1}
    requestBody = request.json
    print(requestBody)
    if request.method == "POST":
        if requestBody["modelType"]==1:
            data["predict"] = int(model_predict_type_one(requestBody["imagePath"]))
            data["success"] = True
        if requestBody["modelType"]==2:            
            data["predict"] = int(model_predict_type_two(requestBody["imagePath"]))
            data["success"] = True
        print(data)
        return flask.jsonify(data)

if __name__ == '__main__':
    api.run() 
