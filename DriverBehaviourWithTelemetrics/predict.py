from __future__ import print_function

import argparse
import pandas as pd
import numpy as np
from numpy.random import randint
import math
import csv
import json

from copy import deepcopy
from joblib import load


class StackingRegressor:
    def __init__(self, models, second_model, features):
        self.models = models
        self.feature_models = []
        self.second_model = second_model
        self.features = features

    def _generate_f_features(self, X):
        f_features = np.zeros((X.shape[0], len(self.features) * len(self.models)))
        for num, features in enumerate(self.features * len(self.models)):
            model = self.feature_models[num]
            f_features[:, num] = model.predict(X.loc[:, features[1]])
        return f_features

    def fit(self, X, y):
        # generate multiple trained models with different features
        for model in self.models:
            for feature in self.features:
                model.fit(X.loc[:, feature[1]], y)
                self.feature_models.append(deepcopy(model))
        f_features = self._generate_f_features(X)
        self.second_model.fit(f_features, y)

    def predict(self, X):
        f_features = self._generate_f_features(X)
        return self.second_model.predict(f_features)

    def predictfn(filepath):
        sr = load("./model/stackingRegressor.joblib")
        test_data = pd.read_csv(filepath)
        shape = test_data.shape
        print(shape)
        testing_ID = randint(2000, 5000)
        test_data['Accuracy'] = randint(3, 12, shape[0])
        test_data['Bearing'] = randint(0, 355, shape[0])
        test_data['Testing_ID'] = testing_ID
        test_data = test_data.rename({'placement_x': 'gyro_x', 'placement_y': 'gyro_y','placement_z': 'gyro_z'}, axis='columns')

        detailed_trace = []
        i = 0
        j = 0
        previous = ''
        current  = ''
        for val in test_data['path'].values:
            if i== 0:
                previous = val
                current = val
                detailed_trace.insert(j, {'path_value': current, 'second_value': int(test_data['second'].values[i])})
            elif i > 0:
                previous = current
                current = val
                if current == previous:
                    i = i + 1
                    continue
                else:
                    j = j + 1
                    detailed_trace.insert(j, {'path_value': current, 'second_value': int(test_data['second'].values[i])})
            i = i + 1
        j = j + 1
        detailed_trace.insert(j, {'path_value': current, 'second_value': int(test_data['second'].values[i-1])})
        print(detailed_trace)

        test_data.drop(columns=["path"], inplace=True)
        test_X = pd.DataFrame()

        for col in test_data.columns:
            if col != "Testing_ID":
                temp = test_data.groupby("Testing_ID")[col].agg(["mean", "sum", "max", "min"])
                test_X[col + "_mean"] = temp["mean"]
                test_X[col + "_sum"] = temp["sum"]
                test_X[col + "_max"] = temp["max"]
                test_X[col + "_min"] = temp["min"]
        print(test_X)
        test_X = test_X.reset_index(drop=True)
        test_X.drop(columns=["second_min"], inplace=True)

        # generate distance, velocity and angle features
        for col in test_X.columns:
            if col.startswith("second"):
                agg_method = col.split("_")[1]
                test_X["distance_" + agg_method] = test_X[col] * test_X["Speed_" + agg_method]
                test_X["velocity_x_" + agg_method] = test_X[col] * test_X["acceleration_x_" + agg_method]
                test_X["velocity_y_" + agg_method] = test_X[col] * test_X["acceleration_y_" + agg_method]
                test_X["velocity_z_" + agg_method] = test_X[col] * test_X["acceleration_z_" + agg_method]
                test_X["angle_x_" + agg_method] = test_X[col] * test_X["gyro_x_" + agg_method]
                test_X["angle_y_" + agg_method] = test_X[col] * test_X["gyro_y_" + agg_method]
                test_X["angle_z_" + agg_method] = test_X[col] * test_X["gyro_z_" + agg_method]

        y_pred = sr.predict(test_X)
        result = {"prediction": float("{:.4f}".format(y_pred[0])),"speed_max": float("{:.3f}".format(test_X["Speed_max"][0])), "second_max": int(test_X["second_max"][0]), "total_minutes":math.ceil(test_X["second_max"] / 60), 'detail_result':detailed_trace }
        high_level = {"prediction": str(float("{:.4f}".format(y_pred[0]))),"speed_max": str(float("{:.3f}".format(test_X["Speed_max"][0]))), "second_max": str(test_X["second_max"][0]), "total_minutes":str(math.ceil(test_X["second_max"] / 60))}  # json.dumps take a dictionary as input and returns a string as output.
        res_1 = json.loads(json.dumps(high_level))
        with open('./result/TestResult_' + str(testing_ID) + '.csv', "w") as f:
            writer = csv.writer(f)
            for i in res_1:
                writer.writerow([i, res_1[i]])
        f.close()
        return result
        # return {'high_result': result, 'detail_result':detailed_trace}