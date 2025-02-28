import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import Dense, Input, Dropout, LeakyReLU, Activation #type: ignore
from tensorflow.keras.optimizers import Adam #type: ignore
from tensorflow.keras.callbacks import EarlyStopping #type: ignore
from datetime import datetime, timedelta

# Define the file path
file_path = 'codex\cleaned_Switzerland_Raw_Data.xlsx'

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

# Build the neural network model
model = Sequential()
model.add(Input(shape=(X_train.shape[1],)))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile the model with a slightly reduced learning rate
model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')

# Define the EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# Train the model with the callback
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=300, batch_size=64, callbacks=[early_stopping])

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate error metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Function to calculate Mean Absolute Percentage Error (MAPE)
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Calculate MAPE and accuracy
mape = mean_absolute_percentage_error(y_test, y_pred)
accuracy = 100 - mape

print(f'Test Mean Squared Error: {mse}')
print(f'Test Mean Absolute Error: {mae}')
print(f'Test Mean Absolute Percentage Error: {mape}%')
print(f'Test Accuracy: {accuracy}%')

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