import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, LSTM
from tensorflow.keras import layers
import math
import io # for Model.summery()
import streamlit as st
from contextlib import redirect_stdout

seq = 'A'
algorithm = 'MLP'
tema = '전력'

st.title('전력사용량 MLP 알고리즘 적용')

data = np.loadtxt('전력 revenge2.csv',delimiter = ',', dtype = 'object')
data = data[:,:8]
df = pd.DataFrame(data)
#df_without_index = df.reset_index().drop(columns=['index']) ## 사용불가
st.dataframe(df)

D = 3                 # 앞뒤 횟수
##################################################################
x_data = data[1:, 0:4]
y_data = data[1+D:-D , 7:8]  
###################################################################
x_data = x_data.astype('float')
y_data = y_data.astype('float')
################## month ############################################
month = x_data[:, 1:2]
month_1 = np.cos(((month-1)/3)*math.pi)
#month_1 = month
################## high_temp ############################################
high_temp_data1 = data[1:,4:5]
high_temp_data2 = data[1:,5:6]
high_temp_data3 = data[1:,6:7]
high_temp_data1 = high_temp_data1.astype('float')
high_temp_data2 = high_temp_data2.astype('float')
high_temp_data3 = high_temp_data3.astype('float')

high_temp_data1 = (high_temp_data1 - np.mean(high_temp_data1)) / np.std(high_temp_data1)
high_temp_data1 = np.square(high_temp_data1)
high_temp_data2 = (high_temp_data2 - np.mean(high_temp_data2)) / np.std(high_temp_data2)
high_temp_data2 = np.square(high_temp_data2)
high_temp_data3 = (high_temp_data3 - np.mean(high_temp_data3)) / np.std(high_temp_data3)
high_temp_data3 = np.square(high_temp_data3)
high_temp_data = np.concatenate((high_temp_data1,high_temp_data2,high_temp_data3),axis=1)

############################# day 1 #####################################################
day = data[1:,3:4]
day = day.astype('float')
N = day.shape[0]

value = []
for i in range(N-D*2):
    value.append(day[i : i+D*2+1])
    day_1 = np.array(value)
day_1 = np.array(day_1)
#### 
st.info(f"day1 Shape : {day_1.shape}")
day_1 = day_1.reshape(day_1.shape[0], D*2+1)
day_1 = day_1.sum(axis=1)

day_1 = day_1.reshape(-1,1)
day = data[1:,3:4]
day = day.astype('float')
N = day.shape[0]
######################### day 2 #########################################################
d = 1          
value = []
for i in range(N-d):
    value.append(day[i : i+d+1])
    day_4 = np.array(value)

day_4 = np.array(day_4)
day_4 = day_4.reshape(day_4.shape[0], 2)
day_4 = day_4.sum(axis=1)
day_4 =day_4.reshape(-1,1)
dayday = day[1:]
value = []
for i in range(day_4.shape[0]):
    value.append(day_4[i:i+1] * dayday[i:i+1])    
day_4 = np.array(value)
day_4 = np.squeeze(day_4, axis=1)
day_4 = day_4[D-1:-D,:]

##################################################################################
x_data = np.concatenate((x_data[D:-D,:],month_1[D:-D,:], -day_1, -day_4),axis=1)
###############################################################################
variable = x_data.shape[1]
x = []
for i in range(variable):
    x_data1 = x_data[:, i:i+1]
    x_norm = (x_data1 - np.mean(x_data1)) / np.std(x_data1)
    x.append(x_norm)

x = np.array(x)
x = np.squeeze(x, axis = 2)
x = np.transpose(x)
############################################################################    
x_norm = np.concatenate((x,high_temp_data[D:-D,:]),axis=1)
y_data = (y_data - np.mean(y_data)) / np.std(y_data)
#############################################################################
#### st.write(x_norm.shape)
st.warning(f"XNorm shape : {x_norm.shape}")

per = 0.6
per_2 = 0.1
train_shape = int(len(y_data)*per)
valid_shape = int(len(y_data)*(per+per_2))
X_train = x_norm[:train_shape,:]
X_test  = x_norm[valid_shape:,:]
X_valid = x_norm[train_shape:valid_shape,:]

y_train = y_data[:train_shape,:]
y_valid = y_data[train_shape:valid_shape,:]
y_test  = y_data[valid_shape:,:]

#st.write(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
st.error(f"X_train.shape: {X_train.shape} y_train.shape: {y_train.shape}, X_test.shape:{X_test.shape} y_test.shape:{y_test.shape}")

model = Sequential()
model.add(Dense(20, input_shape=(X_train.shape[1],)))
model.add(Dense(20))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

#########################  For Summary()
buf = io.StringIO()

with redirect_stdout(buf):
    model.summary()

st.text(buf.getvalue())

#fit = model.fit(X_train, y_train, nb_epoch= 500 ,validation_data=(X_valid,y_valid))
fit = model.fit(X_train, y_train, epochs= 100 ,validation_data=(X_valid,y_valid))
prediction1 = model.predict(X_train)
prediction2 = model.predict(X_test)

plt.figure(figsize=(20, 5))
plt.plot(fit.history['loss'][3:], 'r')
plt.ylabel('loss')
plt.xlabel('Epoch')
st.pyplot(plt)
#plt.legend()

n  = '5'
fig1 = plt.figure(figsize=(10, 5))
plt.plot(prediction1, label = 'train_pred')
plt.plot(y_train, label = 'observ')
plt.legend()
plt.title("train",  fontsize=15)
st.pyplot(fig1) # 첫 번째 그래프를 그립니다.

fig2 = plt.figure(figsize=(10, 5))
plt.plot(prediction2, label = 'test_pred')
plt.plot(y_test, label = 'observ')
plt.legend()
plt.title("test",  fontsize=15)
st.pyplot(fig2) # 두 번째 그래프를 그립니다.
date_range = pd.date_range("2014-1-4", "2018-12-28")
pred1 = np.concatenate([prediction1, prediction2], axis=0)

x = 1000
y = x+200
plt.figure(figsize=(20, 5))
plt.plot(pred1[x:y] ,label = 'predict')
plt.plot(y_data[x:y], label = 'observ')
plt.legend(loc ="best")
plt.title("test 0 ~ 200", fontsize = 15)
st.pyplot(plt)

# In[36]:
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(prediction2,y_test))

st.write("RMES & MAPE")
y_data1 = data[1: , 7:8]
y_data1 = y_data1.astype('float')

def MAPE(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def return_std(data):

    data = np.array(data)
    return data * np.std(y_data1) + np.mean(y_data1)

st.info(f"MAPE : {MAPE(return_std(y_test),return_std(prediction2))}")
st.success(f"RMES :{math.sqrt(mean_squared_error(return_std(y_test),return_std(prediction2)))}")





