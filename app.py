from datetime import datetime,timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import streamlit as st
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
yf.pdr_override()
#df.head()
#print(datetime.today().strftime('%Y-%m-%d'))
start='2012-01-01'
end=str(datetime.today().strftime('%Y-%m-%d'))
tomorrow = datetime.now() + timedelta(1)
st.title('Stock Trend Prediction')
user_input=st.text_input('Enter stock ticker','AAPL')

#df=data.DataReader(user_input,'stooq',start,end)
df=yf.download(user_input,start,end)
df=df.reset_index()
df=df.drop('Date',axis=1)
st.subheader(f'Data from 2012-01-01 to {end}')
st.write(df.describe())
#visualisation
st.subheader('Closing Price vs Time Chart')
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

#with ma100
st.subheader('Closing Price vs Time Chart with 100MA')
ma100=df.Close.rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

#with ma200
st.subheader('Closing Price vs Time Chart with 200MA & 100MA')
ma200=df.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)

#splitting the data into training and testing
data_train=pd.DataFrame(df['Close'][0:int(len(df)*0.7)])
data_testing=pd.DataFrame(df['Close'][int(len(df)*0.7):int(len(df))])
#print(data_train.shape,data_testing.shape)

scaler=MinMaxScaler(feature_range=(0,1))
data_training_array=scaler.fit_transform(data_train)

#load the model
model = load_model('model.h5')

past_100_days=data_train.tail(100)
final_df=past_100_days.append(data_testing,ignore_index=True)
input_data=scaler.fit_transform(final_df)
x_test=[]
y_test=[]
for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
x_test,y_test=np.array(x_test),np.array(y_test)
y_predicted=model.predict(x_test)
scalr=scaler.scale_
scale_factor=1/scalr[0]
y_predicted=y_predicted*scale_factor
#print(y_predicted)
y_test=y_test*scale_factor
st.subheader('Prediction vs original')
fig2=plt.figure(figsize=(12,6))

plt.plot(y_test,'b',label='Original Price')
plt.plot(y_predicted,'r',label="predicted price")
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
#print("tu joothi",y_test[-1])
cx=np.array([x_test[-1]])
p=model.predict(cx)
tomorrow= tomorrow.strftime('%d-%m-%Y')
st.subheader(f'Closing Price predicted for {tomorrow} is:{p*scale_factor}')