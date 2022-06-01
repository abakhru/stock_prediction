#!/usr/bin/env python

from explainx import *
import xgboost

x_data, y_data = explainx.dataset_boston()
model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(x_data, label=y_data), 100)

explainx.ai(x_data, y_data, model, model_name="xgboost")


