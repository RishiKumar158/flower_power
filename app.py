from flask import Flask, render_template, url_for, request, session, redirect
from flask_cors import cross_origin
import numpy as np
import joblib


def return_prediction(model, sample):
    s_len = sample['sepal_length']
    s_wid = sample['sepal_width']
    p_len = sample['petal_length']
    p_wid = sample['petal_width']

    flower = [[s_len, s_wid, p_len, p_wid]]

    flower_classes = np.array(['setosa', 'versicolor', 'virginica'])

    class_ind = model.predict(flower)

    return flower_classes[class_ind][0]


app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = "Rishi's Flower_Power"
model = joblib.load('flower_power.joblib')


@app.route('/', methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
@cross_origin()
def predict():
    if request.method == 'POST':
        s_len = float(request.form['sepal_length'])
        s_wid = float(request.form['sepal_width'])
        p_len = float(request.form['petal_length'])
        p_wid = float(request.form['petal_width'])

        flower = [[s_len, s_wid, p_len, p_wid]]

        flower_classes = np.array(['setosa', 'versicolor', 'virginica'])

        class_ind = model.predict(flower)

        if class_ind == 0:
            return render_template('setosa.html', s_len=s_len, s_wid=s_wid, p_len=p_len, p_wid=p_wid)
        elif class_ind == 1:
            return render_template('versicolor.html', s_len=s_len, s_wid=s_wid, p_len=p_len, p_wid=p_wid)
        else:
            return render_template('virginica.html', s_len=s_len, s_wid=s_wid, p_len=p_len, p_wid=p_wid)
    return render_template('predictor.html')


if __name__ == '__main__':
    app.run()
