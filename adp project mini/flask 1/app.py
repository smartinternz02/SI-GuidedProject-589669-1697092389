import numpy as np
import os
from keras.preprocessing import image
import pandas as pd
import cv2
import tensorflow as tf
# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.python.keras.backend import set_session 
from tensorflow.python.keras.models import load_model

# Import BatchNormalization from the correct module
from tensorflow.keras.layers import BatchNormalization

# Enable eager execution after importing TensorFlow
tf.compat.v1.disable_eager_execution()
sess = tf.compat.v1.Session()

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
graph = tf.get_default_graph()
app = Flask(__name__)
set_session(sess)

# Define a custom object for BatchNormalization layer
custom_objects = {'BatchNormalization': BatchNormalization}

# Load the model weights with custom_objects
model = load_model('adp.h5')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict1():
    return render_template('alzpre.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from the post request
        f = request.files['image']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        img = image.load_img(file_path, target_size=(224,224))
        x = image.img_to_array(img)
        X = np.expand_dims(x, axis=0)
        with graph.as_default():
            set_session(sess)
            prediction = model.predict(X)[0][0][0]
        print(prediction)
        #if prediction == 0:
           # text = "Mild Demented"
        #elif prediction == 1:
            #text = "Moderate Demented"
        #elif prediction == 2:
            #text = "Non Demented" 
        #else:
            #text = "Very Mild Demented"
        #return text

if __name__ == "__main__":
    app.run(debug=True)






