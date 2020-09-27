from flask import Flask
from flask_cors import CORS
from flask import request, Response
import tensorflow as tf
import numpy as np
import time

app = Flask(__name__)
cors = CORS(app, support_credentials=True, resources={r"/api/*": {"origins": "*", "support_credentials": True}})




@app.route('/api/hw', methods=['GET'])
def hello_world():
    print("Hello, World!")
    return "Hello, World!"



@app.route('/api/select_model/', methods=['POST'])
def select_model():
    json_data = request.get_json()

    model = json_data['model_name']
    tf.keras.models.load_model("C:/Users/marku/Desktop/gan_training_output/perceptual/sw_0.00000000001_cw_0.001/20k/celeba/41735/generator1")



if __name__ == '__main__':
    app.run()