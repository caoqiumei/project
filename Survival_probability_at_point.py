#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
from sklearn import svm
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import set_config
from sklearn.preprocessing import OrdinalEncoder
from sklearn.inspection import permutation_importance
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv #转换格式


# In[12]:





def OS_Survival_probability_at_point(X,rsf_os):
    surv1 = rsf_os.predict_survival_function(X, return_array=True)#生存概率预测
    list1=[]
    for i in rsf_os.event_times_:
        a=str(i)
        list1.append(a)
    pdd_o=pd.DataFrame(surv1,columns=list1)
    pdd_o['os_5_pre']=pdd_o[['5.0','5.08','5.25','5.42','5.5','5.67','5.75','5.83','5.92']].mean(axis=1)
    pdd_o['os_10_pre']=pdd_o[['10.0','10.08','10.17','10.25','10.33','10.42','10.67','10.75','10.83']].mean(axis=1)
    pdd_o['os_20_pre']=pdd_o[['20.0','20.17','20.33','20.42','20.5','20.58','20.67','20.75','20.83','20.92']].mean(axis=1)
    pd_os=pdd_o[['os_5_pre','os_10_pre','os_20_pre']]
    print(pd_os)

    


# In[13]:




def CSS_Survival_probability_at_point(X,rsf_css):
    surv2 = rsf_css.predict_survival_function(X, return_array=True)#生存概率预测
    list2=[]
    for i in rsf_css.event_times_:
        a=str(i)
        list2.append(a)
    pdd=pd.DataFrame(surv2,columns=list2)
    pdd['css_5_pre']=pdd[['5.25','5.42','5.5','5.67','5.83','5.92']].mean(axis=1)
    pdd['css_10_pre']=pdd[['10.17','10.42','10.67']].mean(axis=1)
    pdd['css_20_pre']=pdd[['20.5','20.75','20.92']].mean(axis=1)
    pdd_css=pdd[['css_5_pre','css_10_pre','css_20_pre']]
    print(pdd_css)




    

