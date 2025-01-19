import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import Dense, Input, Dropout #type:ignore
from tensorflow.keras.optimizers import Adam #type: ignore
from tensorflow.keras.callbacks import EarlyStopping #type: ignore
from datetime import datetime, timedelta

# Define the file path
file_path = 'data/model.xlsx'

# Check if the file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file {file_path} does not exist. Please check the path.")

# Load data from Excel workbook using openpyxl
training_data = pd.read_excel(file_path, sheet_name='training_data', engine='openpyxl')
validation_data = pd.read_excel(file_path, sheet_name='validation_data', engine='openpyxl')
testing_data = pd.read_excel(file_path, sheet_name='testing_data', engine='openpyxl')

# Print the columns to check if 'target' exists
print("Training Data Columns:", training_data.columns)
print("Validation Data Columns:", validation_data.columns)
print("Testing Data Columns:", testing_data.columns)

# Trim any whitespace from the column names
training_data.columns = training_data.columns.str.strip()
validation_data.columns = validation_data.columns.str.strip()
testing_data.columns = testing_data.columns.str.strip()

# Check for NaN or infinite values
print("Checking for NaN or infinite values...")
print(training_data.isna().sum())
print(validation_data.isna().sum())
print(testing_data.isna().sum())

# Drop rows with NaN or infinite values
training_data = training_data.replace([np.inf, -np.inf], np.nan).dropna()
validation_data = validation_data.replace([np.inf, -np.inf], np.nan).dropna()
testing_data = testing_data.replace([np.inf, -np.inf], np.nan).dropna()

# Prepare the data
X_train = training_data.drop(['Actual Total Load', 'Time', 'Date'], axis=1)
y_train = training_data['Actual Total Load']
X_val = validation_data.drop(['Actual Total Load', 'Time', 'Date'], axis=1)
y_val = validation_data['Actual Total Load']
X_test = testing_data.drop(['Actual Total Load', 'Time', 'Date'], axis=1)
y_test = testing_data['Actual Total Load']

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Define hyperparameter ranges
neurons = [64, 128, 256, 512]
dropout_rates = [0.2, 0.3, 0.4, 0.5]
learning_rates = [0.001, 0.0005, 0.0001, 0.00005]
batch_sizes = [32, 64, 128]

best_mse = float('inf')
best_mae = float('inf')
best_params = {}

# Grid search for hyperparameters
for neuron in neurons:
    for dropout_rate in dropout_rates:
        for learning_rate in learning_rates:
            for batch_size in batch_sizes:
                # Build the neural network model
                model = Sequential()
                model.add(Input(shape=(X_train.shape[1],)))
                model.add(Dense(neuron, activation='relu'))
                model.add(Dropout(dropout_rate))
                model.add(Dense(neuron // 2, activation='relu'))
                model.add(Dropout(dropout_rate))
                model.add(Dense(neuron // 4, activation='relu'))
                model.add(Dense(1, activation='linear'))

                # Compile the model
                model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')

                # Define the EarlyStopping callback
                early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

                # Train the model with the callback
                model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=batch_size, callbacks=[early_stopping], verbose=0)

                # Evaluate the model
                y_pred = model.predict(X_val)
                mse = mean_squared_error(y_val, y_pred)
                mae = mean_absolute_error(y_val, y_pred)

                # Check if this is the best model
                if mse < best_mse:
                    best_mse = mse
                    best_mae = mae
                    best_params = {
                        'neurons': neuron,
                        'dropout_rate': dropout_rate,
                        'learning_rate': learning_rate,
                        'batch_size': batch_size
                    }

print(f'Best MSE: {best_mse}')
print(f'Best MAE: {best_mae}')
print(f'Best Parameters: {best_params}')

# Build the best model
model = Sequential()
model.add(Input(shape=(X_train.shape[1],)))
model.add(Dense(best_params['neurons'], activation='relu'))
model.add(Dropout(best_params['dropout_rate']))
model.add(Dense(best_params['neurons'] // 2, activation='relu'))
model.add(Dropout(best_params['dropout_rate']))
model.add(Dense(best_params['neurons'] // 4, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile the best model
model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']), loss='mean_squared_error')

# Train the best model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=best_params['batch_size'], callbacks=[early_stopping])

# Evaluate the best model
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate error metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f'Test Mean Squared Error: {mse}')
print(f'Test Mean Absolute Error: {mae}')

# Compare predictions with actual data for the current year
testing_data['Predicted Actual Total Load'] = y_pred
print("Comparison of actual and predicted load for the current year:")
print(testing_data[['Date', 'Time', 'Actual Total Load', 'Predicted Actual Total Load']])

# Generate hourly timestamps for the next year
start_date = datetime.strptime('2024-06-15', '%Y-%m-%d')
end_date = start_date + timedelta(days=365)
hourly_dates = pd.date_range(start=start_date, end=end_date, freq='h')

# Create a DataFrame for future data
future_data = pd.DataFrame({
    'Date': hourly_dates.date,
    'Time': hourly_dates.time,
    'Day-ahead Total Load Forecast': np.random.rand(len(hourly_dates)) * 1000  # Replace with actual forecast if available
})

# Prepare future data
X_future = future_data.drop(['Date', 'Time'], axis=1)
X_future = scaler.transform(X_future)  # Scale future data

# Make predictions for the next year
predictions = model.predict(X_future)
future_data['Predicted Actual Total Load'] = predictions

print("Predictions for the next year:")
print(future_data)