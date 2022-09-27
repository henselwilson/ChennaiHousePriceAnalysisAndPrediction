from django.db import models

# Create your models here.
import pickle
import numpy as np
import xgboost as xgb
from math import floor,ceil
import os
from pathlib import Path


# Create your models here.



class HouseFeatures:
    def __init__(self,**kwargs):
        default_val={
            'sqft': 1200,
            'rooms': 4,
            'btype': 'house',
            'area':'karapakkam',
            'roomsB':2,
            'park': 'yes',
            'street': 'paved',
            'mzzone':'a',
            'age': 15,
            'util' : 'allpub',
            'builtYear': 1997
        }
        default_val.update(kwargs)
        self.sqft=default_val['sqft']
        self.rooms=default_val['rooms']
        self.btype=default_val['btype']
        self.area=default_val['area']
        self.roomsB=default_val['roomsB']
        self.park=default_val['park']
        self.street=default_val['street']
        self.mzzone=default_val['mzzone']
        self.age=default_val['age']
        self.util=default_val['util']
        self.builtYear=default_val['builtYear']



    def predict_val(self):
        with open(os.path.join(Path(__file__).resolve().parent,'XGBoostTreeModel'),'rb') as modelFile:
            regModel=pickle.load(modelFile)
        features=[]
        features.append(int(self.sqft[0]))

        features.append(int(self.rooms[0]))

        if self.btype[0] == "house":
            features.append(0)
            features.append(1)
        elif self.btype[0] == "commercial":
            features.append(1)
            features.append(0)
        else:
            features.append(0)
            features.append(0)

        features.append(int(self.area[0]))

        features.append(int(self.roomsB[0]))

        features.append(int(self.park[0]))

        features.append(int(self.street[0]))

        features.append(int(self.mzzone[0]))

        features.append(int(self.age[0]))

        features.append(int(self.util[0]))

        features.append(int(self.builtYear[0]))

        features=np.array(features,ndmin=2)

        pred_val=regModel.predict(features)

        R = ((pred_val ** 2) * 0.01) ** (0.5)
        Pred_min = int(floor((pred_val - R) / 100000))
        Pred_max = int(ceil((pred_val + R) / 100000))
        result = (Pred_max, Pred_min)
        print(result)
        return result
#featu={'csrfmiddlewaretoken': ['IQmlVS7n1H1qyLxDC66F8ZvHdJLzlzQAO6TWVdBtIb4BCw5RiBSpcPdaT53ji3P2'], 'sqft': ['1000'], 'rooms': ['4'], 'btype': ['house'], 'area': ['0'], 'roomsB': ['2'], 'park': ['1'], 'street': ['0'], 'mzzone': ['0'], 'age': ['12'], 'util': ['0'], 'builtYear': ['2000']}
#newHouse=HouseFeatures(**featu)
#newHouse.predict_val()