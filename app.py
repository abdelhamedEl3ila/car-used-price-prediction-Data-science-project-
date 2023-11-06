from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
      fueltype=request.form.get('fueltype'),
            doornumber=request.form.get('doornumber'),
            carbody=request.form.get('carbody'),
            aspiration=request.form.get('aspiration'),
            symboling=float(request.form.get('symboling')),
            carheight=float(request.form.get('carheight')),
            curbweight=float(request.form.get('curbweight')),
            enginesize=float(request.form.get('enginesize')),
            compressionratio=float(request.form.get('compressionratio')),
            horsepower=float(request.form.get('horsepower')),
            peakrpm=float(request.form.get('peakrpm')),
            highwaympg=float(request.form.get('highwaympg')),
            wheelbase=float(request.form.get('wheelbase'))


        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)        