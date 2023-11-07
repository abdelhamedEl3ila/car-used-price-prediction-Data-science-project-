import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path='artifacts\model.pkl'
            preprocessor_path='artifacts\proprocessor.pkl'
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        fueltype: str,
        doornumber: str,
        carbody: str,
        cylindernumber: str,
        aspiration: str,
        symboling: float,
        carheight: float,
        curbweight: float,
        enginesize: float,
        compressionratio: float,
        horsepower: float,
        peakrpm:int,
        wheelbase:float,
        highwaympg:float

      
        ):

        self.fueltype = fueltype
        self.doornumber = doornumber
        self.carbody = carbody
        self.aspiration = aspiration
        self.cylindernumber = cylindernumber

        self.symboling = symboling
        self.carheight = carheight
        self.curbweight = curbweight
        self.enginesize = enginesize
        self.compressionratio = compressionratio
        self.horsepower = horsepower
        self.peakrpm = peakrpm
        self.highwaympg = highwaympg
        self.wheelbase = wheelbase



    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "fueltype": [self.fueltype],
                "doornumber": [self.doornumber],
                "carbody": [self.carbody],
                "cylindernumber": [self.cylindernumber],
                "aspiration": [self.aspiration],
                "symboling": [self.symboling],
                "carheight": [self.carheight],
                "curbweight": [self.curbweight],
                "enginesize": [self.enginesize],
                "compressionratio": [self.compressionratio],
                "horsepower": [self.horsepower],
                "peakrpm": [self.peakrpm],
                "highwaympg": [self.highwaympg],
                "wheelbase": [self.wheelbase]



            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
