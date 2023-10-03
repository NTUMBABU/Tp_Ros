import pickle
from flask import Flask, request, jsonify, render_template
import pandas as pd
import seaborn as sns
import numpy as np

#create flask app
app = Flask(__name__)

#load the pickel model
modelend = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_feature = [np.array(int_features)]
    prediction = modelend.predict(final_feature)

    output = round(prediction[0],2)

    return render_template('index.html', prediction_text='vous avez {}'.format(output))

@app.route('/results', methods=['POST'])
def results():
    data = request.get_json(force=True)
    prediction = modelend.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)