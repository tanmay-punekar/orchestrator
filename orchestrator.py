# Mathematical functions 
import math 
# Fundamental package for scientific computing with Python
import numpy as np 
# Additional functions for analysing and manipulating data
import pandas as pd 
# Date Functions
from datetime import date, timedelta, datetime
# This function adds plotting functions for calender dates
from pandas.plotting import register_matplotlib_converters
# Important package for visualization - we use this to plot the market data
import matplotlib.pyplot as plt 
# Formatting dates
import matplotlib.dates as mdates
# Packages for measuring model performance / errors
from sklearn.metrics import mean_absolute_error, mean_squared_error
# Deep learning library, used for neural networks
from keras.models import Sequential 
# Deep learning classes for recurrent and regular densely-connected layers
from keras.layers import LSTM, Dense, Dropout
# EarlyStopping during model training
from keras.callbacks import EarlyStopping
from keras.layers.recurrent import SimpleRNN

#DATA CONCATENATION

df1 = pd.read_csv(r"C:\Users\Tanmay\Desktop\Case_Study_1\FBcomments.csv",encoding = "ISO-8859-1")
df1["Date"] = pd.to_datetime(df1["Date"])
df1['Comment'] = df1['Comment'].astype(str)

df2 = pd.read_csv(r"C:\Users\Tanmay\Desktop\Case_Study_1\Insta1.csv",encoding = "ISO-8859-1")
df2["Date"] = pd.to_datetime(df2["Date"])
df2['Comment'] = df2['Comment'].astype(str)

df3 = pd.read_csv(r"C:\Users\Tanmay\Desktop\Case_Study_1\Youtube1.csv",encoding = "ISO-8859-1")
df3["Date"] = pd.to_datetime(df3["Date"])
df3['Comment'] = df3['Comment'].astype(str)

from pymongo import MongoClient
import pandas as pd

client = MongoClient('localhost', 27017)  # Remember your uri string
col = client['casestudy']['twitter2'].find()

df4 = pd.DataFrame(col) 
df4 = df4.rename(columns={"timestamp": "Date"})
df4 = df4.rename(columns={"text": "Comment"})
df5 = df4[['Date','Comment']]
df5["Date"] = pd.to_datetime(df5["Date"])
df5['Comment'] = df5['Comment'].astype(str)
df5['Date'] = df5['Date'].dt.date

odds = pd.read_csv(r"C:\Users\Tanmay\Desktop\Case Study 2\Datasets\odds.csv",encoding = "ISO-8859-1")
manu = manu.merge(odds,how='left', on='Date').fillna('0')

manu = manu.assign(t1=manu.close.shift(-1)).fillna({'t1': manu.close})



#SENTIMENT ANALYSER

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
data['Negative'] = data['Comment'].apply(lambda x: sid.polarity_scores(x)['neg'])
data['Positive'] = data['Comment'].apply(lambda x: sid.polarity_scores(x)['pos'])
data['Neutral'] = data['Comment'].apply(lambda x: sid.polarity_scores(x)['neu'])
data['Sentiment'] = data['Comment'].apply(lambda x: sid.polarity_scores(x)['compound'])

#TOPIC MODELLER

from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer(max_df = 0.9,min_df = 2,stop_words = 'english')
dtm = cv.fit_transform(data['Comment'].astype(str))
from sklearn.decomposition import NMF
NMF = NMF(n_components=4,random_state=55)
NMF.fit(dtm)
topic_results = LDA.transform(dtm)
df2 = pd.DataFrame(data=topic_results)
data['Topic1'] = df2[0]
data['Topic2'] = df2[1]
data['Topic3'] = df2[2]
data['Topic4'] = df2[3]

#CALCULATION OF FINANCIAL INDICATORS

from finta import TA
manu_df['RSI'] = TA.RSI(manu_df)
manu_df['MFI'] = TA.MFI(manu_df)
manu_df['EMA'] = TA.EMA(manu_df)
manu_df['STOCHK'] = TA.STOCH(manu_df)
manu_df['MACD']= TA.MACD(manu_df)

#FEATURE SCALER

from sklearn.preprocessing import MinMaxScaler
Scaling = MinMaxScaler()
manu[['EMA','MFI','RSI','Positive','Negative','Neutral','Topic1','Topic2','Topic3','Topic4','League','Result','ManBO','OppBO','DrawBO']] = Scaling.fit_transform(manu[['EMA','MFI','RSI','Positive','Negative','Neutral','Topic1','Topic2','Topic3','Topic4','League','Result','ManBO','OppBO','DrawBO']])

#MACHINE LEARNING
fi = ['EMA','MFI','RSI']
si = ['Positive','Negative','Neutral']
ti = ['Topic1','Topic2','Topic3','Topic4']
ei = ['League','Result','ManBO','OppBO','DrawBO']
X = manu.loc[:,fi]
y = manu.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 5)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred1 = regressor.predict(X_test)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, y)
y_pred2 = regressor.predict(X_test)

from sklearn.svm import SVR
SVR = SVR()
SVR.fit(X,y)
y_pred3 = SVR.predict(X_test)

from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y_test, y_pred1))
print(mean_absolute_error(y_test, y_pred2))
print(mean_absolute_error(y_test, y_pred3))

from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test, y_pred1))
print(mean_squared_error(y_test, y_pred2))
print(mean_squared_error(y_test, y_pred3))

#DEEPLEARNING

train_dfs = manu
train_df = train_dfs.sort_values(by=['Date']).copy()
date_index = train_df.index

d = pd.to_datetime(train_df.index)
train_df['Month'] = d.strftime("%m") 
train_df['Year'] = d.strftime("%Y") 

# We reset the index, so we can convert the date-index to a number-index
train_df = train_df.reset_index(drop=True).copy()


# Create the dataset with features and filter the data to the list of FEATURES
data = pd.DataFrame(train_df)
data_filtered = data[fi+si+ti]

# We add a prediction column and set dummy values to prepare the data for scaling
data_filtered_ext = data_filtered.copy()
data_filtered_ext['Prediction'] = train_df['t1'] 

nrows = data_filtered.shape[0]
np_data_unscaled = np.array(data_filtered)
np_data_unscaled = np.reshape(np_data_unscaled, (nrows, -1))
print(np_data_unscaled.shape)

# Transform the data by scaling each feature to a range between 0 and 1

# Creating a separate scaler that works on a single column for scaling predictions
np_data = np_data_unscaled
df_Close = pd.DataFrame(train_df['t1'])
np_Close_scaled = df_Close

#RNN

sequence_length = 100

# Split the training data into x_train and y_train data sets
# Get the number of rows to train the model on 70% of the data 
train_data_len = math.ceil(np_data.shape[0] * 0.7) 

# Create the training data
train_data = np_data[0:train_data_len, :]
x_train, y_train = [], []
# The RNN needs data with the format of [samples, time steps, features].
# Here, we create N samples, 100 time steps per sample, and 2 features
for i in range(100, train_data_len):
    x_train.append(train_data[i-sequence_length:i,:]) #contains 100 values 0-100 * columsn
    y_train.append(train_data[i, 0]) #contains the prediction values for validation
    
# Convert the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Create the test data
test_data = np_data[train_data_len - sequence_length:, :]

# Split the test data into x_test and y_test
x_test, y_test = [], []
test_data_len = test_data.shape[0]
for i in range(sequence_length, test_data_len):
    x_test.append(test_data[i-sequence_length:i,:]) #contains 100 values 0-100 * columsn
    y_test.append(test_data[i, 0]) #contains the prediction values for validation
# Convert the x_train and y_train to numpy arrays
x_test, y_test = np.array(x_test), np.array(y_test)

# Convert the x_train and y_train to numpy arrays
x_test = np.array(x_test); y_test = np.array(y_test)
    
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

model = Sequential()

# Model with 100 Neurons 
# inputshape = 100 Timestamps, each with x_train.shape[2] variables
n_neurons = x_train.shape[1] * x_train.shape[2]
print(n_neurons, x_train.shape[1], x_train.shape[2])
model.add(SimpleRNN(n_neurons, return_sequences=False, 
               input_shape=(x_train.shape[1], x_train.shape[2]))) 
model.add(Dense(1, activation='relu'))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

epochs = 5
early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
history = model.fit(x_train, y_train, batch_size=16, 
                    epochs=epochs, callbacks=[early_stop])

predictions = model.predict(x_test)
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y_test, predictions))
from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test, predictions))




#LSTM
sequence_length = 100

# Split the training data into x_train and y_train data sets
# Get the number of rows to train the model on 70% of the data 
train_data_len = math.ceil(np_data.shape[0] * 0.7) 

# Create the training data
train_data = np_data[0:train_data_len, :]
x_train, y_train = [], []
# The RNN needs data with the format of [samples, time steps, features].
# Here, we create N samples, 100 time steps per sample, and 2 features
for i in range(100, train_data_len):
    x_train.append(train_data[i-sequence_length:i,:]) #contains 100 values 0-100 * columsn
    y_train.append(train_data[i, 0]) #contains the prediction values for validation
    
# Convert the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Create the test data
test_data = np_data[train_data_len - sequence_length:, :]

# Split the test data into x_test and y_test
x_test, y_test = [], []
test_data_len = test_data.shape[0]
for i in range(sequence_length, test_data_len):
    x_test.append(test_data[i-sequence_length:i,:]) #contains 100 values 0-100 * columsn
    y_test.append(test_data[i, 0]) #contains the prediction values for validation
# Convert the x_train and y_train to numpy arrays
x_test, y_test = np.array(x_test), np.array(y_test)

# Convert the x_train and y_train to numpy arrays
x_test = np.array(x_test); y_test = np.array(y_test)
    
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

model = Sequential()

# Model with 100 Neurons 
# inputshape = 100 Timestamps, each with x_train.shape[2] variables
n_neurons = x_train.shape[1] * x_train.shape[2]
print(n_neurons, x_train.shape[1], x_train.shape[2])
model.add(LSTM(n_neurons, return_sequences=False, 
               input_shape=(x_train.shape[1], x_train.shape[2]))) 
model.add(Dense(1, activation='relu'))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

epochs = 5
early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
history = model.fit(x_train, y_train, batch_size=16, 
                    epochs=epochs, callbacks=[early_stop])

predictions = model.predict(x_test)
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y_test, predictions))
from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test, predictions))
