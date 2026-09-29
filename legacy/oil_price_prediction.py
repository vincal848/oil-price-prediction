"""SUPERSEDED. Kept for reference; this is the original coursework script.

It does not run against current dependencies - ReservoirPy 0.4 removed
`reservoirpy.verbosity` and the `bias_scaling` / `fb_connectivity` arguments this
file passes - and it has methodological problems that the rewrite documents:

1. MinMaxScaler is fit on the full series before the train/validation split. The
   fitted minimum is -37.63 from 2020-04-20, a date inside the validation set, so
   every training row was normalized by a future value.
2. The reservoir is scored from val[1:] and the LSTM from val[120:], so the two
   models are evaluated over different periods and their error numbers were never
   comparable.
3. There is no naive benchmark. Both models lose to simple persistence, which the
   comparison could not reveal.
4. MAPE is reported on a window containing a negative price.
5. NRMSE divides by the range of the actuals, which the -37.63 print sets.
6. `epochs=6` with `EarlyStopping(patience=10)` can never trigger.
7. The test set is passed as `validation_data` and then scored on.
8. `fb_connectivity=1.1` is a density outside [0, 1], and the model has no
   feedback path for it to apply to.

The working version is data.py, models.py, metrics.py, roll.py and run.py.
"""

import tensorflow as tf
from reservoirpy.nodes import Reservoir, Ridge, ESN
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from time import perf_counter


#Download WTI Data - yfinance module website
data = yf.download("CL=F", start="2000-08-30", end="2024-05-31")

# Isolate Closing Prices + Normalize the data
close_prices = data['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))# Took usage of minmaxscaler from Medium LSTM model article, and class. Had issues with TensorFlow version following an update.
scaled_data = scaler.fit_transform(close_prices)

# Split the data into Training and Validation Sets
train_size = int(len(scaled_data) * 0.75)
train_data = scaled_data[:train_size]
val_data = scaled_data[train_size:]

# Plot the actual price
plt.figure(figsize=(10, 6))
plt.plot(data['Close'])
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('WTI Crude Oil Price per Barrel in Dataset')




############ Reservoir Model ############

res_start = perf_counter()

# Parameters for the reservoir model -
# the selection followed the paper by Kumar K. (2023) [I tested with altered variables, but there was not much advantage changing model specifications]

# Number of Neurons
units = 20

# Leakrate
leakrate = 0.75

spectralradius = 1.025

# Scaling Input
inputscaling = 1.0

# "density" of the reservoir matrix
rcconnectivity = 0.15

# Reservoir Input value for connecting to the Reservoir
inputconnectivity = 0.2

# Output Connections
fbconnectivity = 1.1

# Regularization Coef for ridge regression
regularizationcoef = 1e-8

# Setting up the Reservoir Layer from ReservoirPy - Written with variables consulting ReservoirPy Module User Guide + Academic Paper.
reservoir = Reservoir(
    units=units,
    sr=spectralradius,
    input_scaling=inputscaling,
    lr=leakrate,
    bias_scaling=1.0,
    input_connectivity=inputconnectivity,
    rc_connectivity=rcconnectivity,
    fb_connectivity=fbconnectivity
)


# Create the output adjustment layer using the Ridge Calculation from ReservoirPy
readout = Ridge(
    ridge=regularizationcoef
    )

# Create an esn using the reservoir and assigning it to the readout connections
esn = reservoir >> readout

# Set up input and output of training by shifting.
train_input = train_data[:-1]
train_output = train_data[1:]

# Fit the esn model onto the training data.
esn.fit(train_input, train_output)

#Set up input and output of validation by shifting.
val_input = val_data[:-1]
val_output = esn.run(val_input)

# Take the inverse transform the predictions
val_output = scaler.inverse_transform(val_output)
actual_output = scaler.inverse_transform(val_data[1:])

res_end = perf_counter()

############ End of Reservoir Model ############




# Plot the Reservoir results

plt.figure(figsize=(10, 6))
plt.plot(data.index[train_size + 1:], actual_output, color='blue', label='Actual Prices')
plt.plot(data.index[train_size + 1:], val_output, color='red', label='Predicted Prices')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Crude Oil Price Prediction using Reservoir Computing')
plt.legend()




############ LSTM Model ############

lstm_start = perf_counter()

# Split the data into training and testing sets (Using same split set as Reservoir)

# Define a function for time steps for a list. - Easier to do this way with the LSTM model. Had some indexing issues with the array.
def create_sequences(data, sequence_length):
    X = []
    y = []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# Setup a timestep consideration for certain period of days - Discussed in the paper:
# I originally selected 60 days, but found lesser error at 120, taking in a period of time into memory.
time_step = 120

# Subset into train and test by list. 
X_train, y_train = create_sequences(train_data, time_step)
X_test, y_test = create_sequences(val_data, time_step)

# Reshape input to be [samples, time steps, features] which is required for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build the LSTM model - Used class material and some reference images/cited google article in consideration of structure.
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=(time_step, 1)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(25))
model.add(tf.keras.layers.LSTM(units=50, return_sequences=False))
model.add(tf.keras.layers.Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Add patience to the model to stop if uneccessary 
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

# Train the model
history = model.fit(X_train, y_train, batch_size=32, epochs=6, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Predictions

# Predict the prices
predicted_prices = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Inverse transform the actual prices
actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

lstm_end = perf_counter()

############ End of LSTM Model ############





# Plot the LSTM results

plt.figure(figsize=(10, 6))
plt.plot(data.index[train_size + time_step:], actual_prices, color='blue', label='Actual Prices')
plt.plot(data.index[train_size + time_step:], predicted_prices, color='red', label='Predicted Prices')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Crude Oil Price Prediction using LSTM')
plt.legend()
plt.show()




############ Calculater Error Metrics ############

# Calculate LSTM Error

def calculate_metrics(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    nrmse = rmse / (np.max(actual) - np.min(actual))
    mape = mean_absolute_percentage_error(actual, predicted)
    return mse, rmse, nrmse, mape

mse, rmse, nrmse, mape = calculate_metrics(actual_prices, predicted_prices)

print(f"LSTM Performance: {lstm_end-lstm_start} seconds")
print(f"LSTM MSE: {mse}")
print(f"LSTM RMSE: {rmse}")
print(f"LSTM NRMSE: {nrmse}")
print(f"LSTM MAPE: {mape}")

# Calculate Reservoir Error

mse, rmse, nrmse, mape = calculate_metrics(actual_output, val_output)

print(f"Reservoir Performance: {res_end - res_start} seconds")
print(f"Reservoir MSE: {mse}")
print(f"Reservoir RMSE: {rmse}")
print(f"Reservoir NRMSE: {nrmse}")
print(f"Reservoir MAPE: {mape}")

print(f"Comparative Performance: {abs((lstm_end-lstm_start)-(res_end - res_start))} seconds difference.")
