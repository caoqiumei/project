# project


# 1.) Instructions for using RFS(random survival forests)

---Save the following files (保存下列文件)

OS(全因死亡):

下载model_os_.z01、model_os_.z02、model_os_.z03、model_os_.zip, 解压model_os_.zip，即可得到model_os.pkl

CSS(特异性死亡): 

下载model_css_.z01、model_css_.zip，解压model_css_.zip，即可得到model_css.pkl

Test file: train.csv

python file: Survival_probability_at_point.py

# 2.) Instuctions for running the code (代码运行说明)

--- In python 3, run the following code (在python 3 环境下，运行下列代码)

1. 安装scikit-survial 包

2. 加载如下python 包

import numpy as np

import pandas as pd

import joblib

from sklearn import set_config

from sklearn.preprocessing import OrdinalEncoder

from sklearn.inspection import permutation_importance

from sksurv.datasets import load_gbsg2

from sksurv.preprocessing import OneHotEncoder

from sksurv.ensemble import RandomSurvivalForest

from sksurv.util import Surv

import Survival_probability_at_point as spap


3. OS全因死亡的预测

x_test=pd.read_csv('x_test.csv') # [Import the test files (载入测试文件)]

xlf_os = joblib.load('model_os.pkl') (载入模型文件)]

xlf_os.predict(x_test) # [Obtain the predictive probability (得到预测值)]

spap.OS_Survival_probability_at_point(x_test,xlf_os)#(得到5年、10年、20年OS的生存概率预测值)


4. CSS特异性死亡的预测

xlf_css = joblib.load('model_css.pkl')

xlf_css.predict(x_test) 

spap.CSS_Survival_probability_at_point(x_test,xlf_css)#(得到5年、10年、20年CSS的生存概率预测值)
