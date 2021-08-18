#!/usr/bin/env python
# coding: utf-8

# # Forecast Exchange Rates

# ## Importing Library

# In[1]:


#VIZ AND DATA MANIPULATION LIBRARY
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

#warnings
import warnings 
warnings.filterwarnings('ignore')

#Library
from sklearn.preprocessing import MinMaxScaler

#Defining the LSTM model
from keras.models import Sequential
from keras.layers import Dense,LSTM
#METRICS
from sklearn.metrics import mean_squared_error

#Streamlit
import streamlit as st 

# ## Loading Data

# In[2]:


inrusd = pd.read_csv('Dataset.csv',parse_dates=["observation_date"])
inrusd.head()


# ### Copy Data

# In[3]:


#Copy Data
df = inrusd.copy()
df.head()


# #### Shape

# In[4]:


df.shape


# #### Renaming 

# In[5]:


#renaming the date and rate
data = df[['observation_date', 'DEXINUS']]
data.columns = ['date', 'rate']


# In[6]:


data.head()


# #### info

# In[7]:


data.info()


# Converting rates to numeric

# In[8]:


data['rate'] = pd.to_numeric(data.rate)


# Sorting Date in Ascending order

# In[9]:


data = data.sort_values('date', ascending=True)


# #### Descriptive Stats

# In[10]:


data.rate.describe()


# #### Checking Null Values

# In[11]:


data.isnull().sum()


# Forward Filling Null Values

# In[12]:


#Forward Filling
data.fillna(method='ffill', inplace=True)


# In[13]:


data.isnull().sum()


# Convert datatype int to float

# In[14]:


#transformation of values to float
data['rate'] = pd.to_numeric(data['rate'], downcast="float")


# #### info

# In[15]:


data.info()


# #### Checking Duplicates


# ## LSTM

# In[32]:


data1 = data.iloc[10000:]


# In[33]:


data2 = data1.copy() 


# In[34]:


print('Start Date:', data2.head(1))
print('---------------------------------')
print('End Date:', data2.tail(1))


# In[35]:


data2.info()


# In[36]:


df1=data2.reset_index()['rate']


# In[37]:


df1


# ### MinMaxScaler

# In[38]:


scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))


# In[39]:


df1


# ### Splitting

# In[40]:


#splitting dataset into train and test split
training_size=int(len(df1)*0.70)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]


# In[41]:


training_size,test_size


# In[42]:


# convertsion of an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)


# In[43]:


# reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)


# In[44]:


print(X_train.shape), print(y_train.shape)


# In[45]:


print(X_test.shape), print(ytest.shape)


# In[46]:


# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


# ### Model

# In[47]:


model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')


# In[48]:


model.summary()


# In[49]:


model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size=64,verbose=1)


# In[50]:


### Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)


# In[51]:


##Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


# #### RMSE

# In[52]:


### Calculate RMSE performance metrics

math.sqrt(mean_squared_error(y_train,train_predict))


# In[53]:


### Test Data RMSE
math.sqrt(mean_squared_error(ytest,test_predict))


# #### Visualization

# In[54]:


### Plotting 
# shift train predictions for plotting
plt.figure(figsize=(20,8))
look_back=100
trainPredictPlot = np.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = np.empty_like(df1)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(df1), label = 'Actual', color = 'g', linewidth=3, linestyle='dotted')
plt.plot(trainPredictPlot, label = 'Actual', color = 'b', linewidth=1)
plt.plot(testPredictPlot, label = 'Actual', color = 'r', linewidth=1)
plt.show()


# In[73]:


len_test_data = len(test_data)
len_test_data


# In[56]:


x_input=test_data[(len_test_data - look_back):].reshape(1,-1)
x_input.shape
#794-100


# In[57]:


temp_input=list(x_input)
temp_input=temp_input[0].tolist()


# In[58]:


temp_input


# ### 30 days future Prediction
#Streamlit start
st.title(" ----------------->Forecast Exchange Rate <----------------")
name=st.text_input("Enter your name here")
day=st.number_input(" Enter Date from 0 to 99 ")

data_obv=float(int(day))
#Streamlit End

# In[59]:


# demonstrate prediction for next 30 days


lst_output=[]
n_steps=100
i=0
while(i<data_obv):
    
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    

print(lst_output)


# In[68]:


day_new=np.arange(1,101)
day_pred=np.arange(101,(101 + data_obv))


# In[69]:


len_val = len(df1)


# #### Visualization

# In[70]:


plt.figure(figsize=(20,8))
plt.plot(day_new,scaler.inverse_transform(df1[(len_val-100):]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))


# In[71]:

'''
plt.figure(figsize=(20,8))
df3=df1.tolist()
df3.extend(lst_output)
plt.plot(df3[(len_val-100):])
'''

# In[72]:


op_res = scaler.inverse_transform(lst_output)
print(op_res)


# In[ ]:
#Streamlit start
if(st.button(" click here for Predict")):
    st.write(op_res)
    st.write('Thanks',name)
#Streamlit end




