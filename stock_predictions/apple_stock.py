#!/usr/bin/env python

import math

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

from stock_predictions import TODAY_DATE, now
from stock_predictions.data_utils import DataUtils

data_utils = DataUtils()
plt.style.use('fivethirtyeight')

STOCK = 'AAPL'

df = data_utils.get_yahoo_stock_data(
    stock_symbol=STOCK,
    start=now.shift(years=-10).format('YYYY-MM-DD'),
    end=now.shift(months=-10).format('YYYY-MM-DD'),
)
# visualize_price_history(df)

# Create a dataframe with only the Close column
data = df.filter(['Close'])
# Convert dataframe into numpy value
dataset = data.values

# Get the number of frows to train the model, using 80% of data for training
training_data_len = math.ceil(len(dataset) * 0.8)

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Create a training dataset
# Create a scaled dataset
train_data = scaled_data[0:training_data_len, :]

# Split the data into x_train and y_train set
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i - 60 : i, 0])
    y_train.append(train_data[i, 0])
    if i <= 60:
        print(x_train)
        print(y_train)
        print()

# Convert the xtrain and ytrain to numpy array
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build LSTM
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train a model
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Creating the testing data
# Creating a new array containing a scaled value from index 1503 to 2003
test_data = scaled_data[training_data_len - 60 :, :]
# Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i - 60 : i, 0])

# convert data into numpy array
x_test = np.array(x_test)

# reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Get model predicted price value
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Get root mean squared error (RMSE)
rmse = np.sqrt(np.mean(predictions - y_test) ** 2)

# plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
# Visualize the data
plt.figure(figsize=(16, 8))
plt.title(f'Prediction Model for {STOCK}')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Actual', 'Predictions'], loc='lower right')
plt.show()

# Get the last 60 day closing price value and covert into dataframe to an array
last_60_days = df[-60:].values
# scale the data to the value 0 to 1
last_60_days_scaled = scaler.transform(last_60_days)
# Create an empty list
X_test = []
# Append the past 60 days
X_test.append(last_60_days_scaled)
# Convert X test into numpy array
X_test = np.array(X_test)
# Reshape the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
# get the predicted scale price
pred_price = model.predict(X_test)
# Undo scaling
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)

# get the quote
apple_quote = data_utils.get_yahoo_stock_data(
    stock_symbol=STOCK, start=now.shift(days=-1).format('YYYY-MM-DD'), end=TODAY_DATE
)
print(apple_quote['Close'])
