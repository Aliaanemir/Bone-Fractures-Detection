from flask import Flask, render_template, request, jsonify, make_response
from flask_cors import CORS
import cv2 
import numpy as np
from tensorflow.keras.models import load_model
import os


# Resizing images 
def images_prep(image_dataset):
    resized = [cv2.resize(img, (250, 250)) for img in image_dataset]
    #normalized = [cv2.normalize(img, None, norm_type=cv2.NORM_MINMAX) for img in resized]
    return np.array(resized) 



# app

APP_ROOT = os.path.dirname(os.path.abspath('Bone-Fracture-API-AWS-P2.py'))

app = Flask(__name__)

app.config['JSON_SORT_KEYS'] = False
CORS(app)

target = os.path.join(APP_ROOT, 'static')
if not os.path.isdir(target):
    os.mkdir(target)

global model_fracture
global model_type

model_fracture = load_model('bone_fracture_baseline_model2.h5')
model_type = load_model('bone_fracture_baseline_model_type.h5')

labels_fracture = ['Negative for Fracture', 'Positive for Fracture']
labels_type = ['ELBOW', 'FINGER', 'FOREARM', 'HAND', 'HUMERUS','SHOULDER', 'WRIST']

@app.route('/predict_fracture', methods=['POST'])
def predict_fracture():
    #save input
    imagefile = request.files['xray']
    image_path = './images' + imagefile.filename
    imagefile.save(image_path)
    
    #preprocess input
    sample= cv2.imread(image_path, 0)
    image = images_prep(sample)  
    
    #prediction
    out = model_fracture.predict(np.reshape(image, (1,250,250,1)))
    max_idx = np.argmax(out,axis=1)
    prediction = labels_fracture[max_idx[0]]

    result = make_response(jsonify({'prediction':prediction}))
   
    return result



@app.route('/predict_type', methods=['POST'])
def predict_type():
    #save input
    imagefile = request.files['xray']
    image_path = './images' + imagefile.filename
    imagefile.save(image_path)
    
    #preprocess input
    sample= cv2.imread(image_path, 0)
    image = images_prep(sample)  
    
    #prediction
    out = model_type.predict(image)
    max_idx = np.argmax(out,axis=1)
    prediction = labels_type[max_idx[0]]

    result = make_response(jsonify({'prediction':prediction}))
   
    return result

@app.route('/predict_diagnosis', methods=['POST'])
def predict_diagnosis():
    #save input
    imagefile = request.files['xray']
    image_path = './images' + imagefile.filename
    imagefile.save(image_path)
    
    #preprocess input
    sample= cv2.imread(image_path, 0)
    image = images_prep(sample)  
    
    #prediction
    out_region = model_type.predict(image)
    max_idx = np.argmax(out_region,axis=1)
    accuracy_reg = out_region.max()
    print(accuracy_reg)
    prediction_region = labels_type[max_idx[0]]
    
    out_fracture = model_fracture.predict(image)
    print(out_fracture)
    max_idx = np.argmax(out_fracture,axis=1)
    accuracy_fracture = out_fracture.max()
    prediction_fracture = labels_fracture[max_idx[0]]

    result_dict = {'MostProbably':[{'Status':prediction_fracture, 'Accuracy': str(round((accuracy_fracture),3))}, {'Region':prediction_region, 'Accuracy': str(round((accuracy_reg),3))}]}

    result = make_response(jsonify((result_dict)))
   
    return result
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000)


# In[ ]:




