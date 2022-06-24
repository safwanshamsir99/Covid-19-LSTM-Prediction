# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 09:26:06 2022

@author: safwanshamsir99
"""

import os
import pandas as pd
import numpy as np
import pickle
import datetime

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import plot_model

from module_covid19 import EDA,ModelCreation,ModelEvaluation

#%% STATIC
CSV_PATH = os.path.join(os.getcwd(),'cases_malaysia_train.csv')
CSV_TEST_PATH = os.path.join(os.getcwd(),'cases_malaysia_test.csv')
MMS_PATH = os.path.join(os.getcwd(),'mms_covid_cases.pkl')
log_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
LOG_FOLDER_PATH = os.path.join(os.getcwd(),'logs',log_dir)

#%% DATA LOADING
df = pd.read_csv(CSV_PATH,na_values='?')

df['cases_new'] = pd.to_numeric(df['cases_new'],errors='coerce') 
# to change the data type into float64 and convert blank space into NaNs

#%% DATA INSPECTION
df.info()
stats = df.describe().T
df.isna().sum() # 12 NaNs(cases_new), 342 NaNs(cluster column)
df.duplicated().sum() # 0 duplicate data (of course zero haha)

eda = EDA()
eda.plot_graph(df) # to plot the graph

#%% DATA CLEANING 
df['cases_new'] = df['cases_new'].interpolate()# acts like fillna for time series
df['cases_new'] = np.ceil(df['cases_new']) # to complete 1 body count, body count cannot be in float
df.isna().sum()
df.info()
'''
Only interpolate cases_new column, and didnt interpolating the cluster columns 
since it will not be used as features
'''

#%% FEATURES SELECTION
# selecting only cases_new column.

#%% PREPROCESSING
# Scaling process
mms = MinMaxScaler()
df = mms.fit_transform(np.expand_dims(df['cases_new'],axis=-1))

# save using pickle
with open(MMS_PATH,'wb') as file:
    pickle.dump(mms,file)

X_train = []
y_train = []
win_size = 30

for i in range(win_size,np.shape(df)[0]):
    X_train.append(df[i-win_size:i,0])
    y_train.append(df[i,0])
    
X_train = np.array(X_train)
y_train = np.array(y_train)
# never perform train test split for time series data

#%% MODEL DEVELOPMENT
mc = ModelCreation()
model = mc.simple_lstm_layer(X_train)

plot_model(model,show_layer_names=(True),show_shapes=(True))

X_train = np.expand_dims(X_train,axis=-1)

# CALLBACKS
tensorboard_callbacks = TensorBoard(log_dir=LOG_FOLDER_PATH)

hist = model.fit(X_train,y_train,batch_size=32,epochs=1000,
                 callbacks=(tensorboard_callbacks))

#%% MODEL EVALUATION
hist.history.keys()

me = ModelEvaluation()
me.plot_model_evaluation(hist)

#%% MODEL ANALYSIS
test_df = pd.read_csv(CSV_TEST_PATH)
test_df['cases_new'] = test_df['cases_new'].interpolate()
test_df = mms.transform(np.expand_dims(test_df['cases_new'].values,axis=-1))
con_test = np.concatenate((df,test_df),axis=0)
con_test = con_test[-130:] # 30(win_size) + 100(test_df)

X_test = []
for i in range(win_size,len(con_test)):
    X_test.append(con_test[i-win_size:i,0])
  
# Another code for line above
# X_test = [con_test[i-win_size:i,0] for i in range(win_size,len(con_test))]

X_test = np.array(X_test)

predicted = model.predict(np.expand_dims(X_test,axis=-1))

#%% plotting the graph
me.plot_predicted_graph(test_df, predicted, mms)

#%% MSE, MAPE
test_df_inversed = mms.inverse_transform(test_df)
predicted_inversed = mms.inverse_transform(predicted)

print('mae: ' + str(mean_absolute_error(test_df,predicted)))
print('mse: ' + str(mean_squared_error(test_df,predicted)))
print('mape: ' + str(mean_absolute_percentage_error(test_df,predicted)))

print('mae_i: ' + str(mean_absolute_error(test_df_inversed,predicted_inversed)))
print('mse_i: ' + str(mean_squared_error(test_df_inversed,predicted_inversed)))
print('mape_i: ' + str(mean_absolute_percentage_error(test_df_inversed,
                                                      predicted_inversed)))

#%% DISCUSSION/REPORTING
'''
The model is able to predict the trend of the Covid-19 cases in Malaysia.

Mean absolute error(MAE) and mean squared error(MSE) report 4.18% and 0.47%
respectively when tested using the testing dataset.

Based on the Loss graph displayed using matpotlib, loss occured during
the dataset training is nearly 0% with a high amount of epochs (1000).

The deep learning model used only 3 layers; input layer, LSTM layer and 
output layer. The number of nodes is set to 15, and the dropout rate is 
set to 0.03. Rectified linear unit (ReLU) is used as an activation function.

Based on the mean absolute percentage error(MAPE) which around 0.08%, 
this model can be considered as successful as it can predicted the trendline 
of Covid-19 cases in Malaysia.
'''
























