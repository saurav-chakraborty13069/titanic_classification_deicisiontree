from flask import Flask, render_template, jsonify, request
from flask_cors import cross_origin
from train import train_data
from predict import predict_data
from logger import App_Logger
import os

app = Flask(__name__)

log_writer = App_Logger()

@app.route('/',methods = ['GET'])
@cross_origin()
def home_page():
    return render_template('index.html')

@app.route('/train',methods = ['GET','POST'])
@cross_origin()
def train():
    train_data(log_writer)


    return render_template('index.html')


@app.route('/predict',methods = ['GET','POST'])
@cross_origin()
def predict():
    if request.method == 'POST':
        try:
            file_object = open("logs/GeneralLogs.txt", 'a+')
            log_writer.log(file_object, 'Start getting data from UI')
            Pclass = float(request.form['Pclass'])
            Sex = (request.form['Sex'])
            Age = float(request.form['Age'])
            SibSp = float(request.form['SibSp'])
            Parch = float(request.form['Parch'])
            Fare = float(request.form['Fare'])


            log_writer.log(file_object, 'Complete getting data from UI')

            mydict = {'Pclass': Pclass, 'Sex': Sex,'Age': Age, 'SibSp': SibSp,
                      'Parch': Parch, 'Fare': Fare }

            log_writer.log(file_object, 'Passing mydict to prediction.predict_data')
            prediction = predict_data(mydict, log_writer)

            return render_template('results.html', prediction=prediction)
            log_writer.log(file_object, '=================================================')

        except Exception as e:
            print('The Exception message is: ', e)
            return 'something is wrong'
        # return render_template('results.html')
    else:
        return render_template('index.html')





if __name__ == '__main__':
    app.run(host = '127.0.0.1', port = 8001, debug=True)

