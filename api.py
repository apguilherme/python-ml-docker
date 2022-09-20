# run: python api.py

from flask import Flask, request
from flasgger import Swagger
import pickle
import pandas as pd
import numpy as np

with open("./pickles/forest.pkl", "rb") as model_file: # model to predict from
    model = pickle.load(model_file)

app = Flask(__name__)
swagger = Swagger(app) # http://127.0.0.1:5050/apidocs/

@app.route("/iris", methods=["GET"]) # http://127.0.0.1:5050/iris?s_length=5.7&s_width=5.6&p_length=4.3&p_width=7.8
def iris():
    """
    Returns a prediction of iris. 
    ---
    parameters:
        - name: s_length
          in: query
          type: number
          required: true
        - name: s_width
          in: query
          type: number
          required: true
        - name: p_length
          in: query
          type: number
          required: true
        - name: p_width
          in: query
          type: number
          required: true
    responses:
        200:
            description: OK
    """
    s_length = request.args.get("s_length")
    s_width = request.args.get("s_width")
    p_length = request.args.get("p_length")
    p_width = request.args.get("p_width")
    prediction = model.predict(np.array([[s_length, s_width, p_length, p_width]]))
    return str(prediction)

@app.route("/iris_file", methods=["POST"]) # http://127.0.0.1:5050/iris_file
def iris_input():
    """
    Returns a prediction of iris given a csv file. 
    ---
    parameters:
        - name: input_file
          in: formData
          type: file
          required: true
    responses:
        200:
            description: OK
    """
    input_file = pd.read_csv(request.files.get("input_file"), header=None) # form-data
    prediction = model.predict(input_file)
    return str(prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)

