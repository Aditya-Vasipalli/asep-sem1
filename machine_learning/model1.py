import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from datetime import datetime, timedelta

# Define the folder path containing the Excel files
folder_path = './codex'  # Use relative path

# Define the date range for predictions
start_date = '2025-01-01'
end_date = '2026-01-01'

# Custom callback for early stopping if error is above 50 after a certain number of epochs
class EarlyStoppingByError(Callback):
    def __init__(self, start_epoch=1800):
        super(EarlyStoppingByError, self).__init__()
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self.start_epoch and logs.get('val_loss') > 50:
            print(f"\nEpoch {epoch}: Early stopping as validation loss is above 50")
            self.model.stop_training = True

# Function to preprocess the data
def preprocess_data(df):
    # Check if 'Date' and 'Time' columns exist
    if 'Date' not in df.columns or 'Time' not in df.columns:
        raise KeyError("The required columns 'Date' and 'Time' are not present in the DataFrame.")
    
    # Convert Date column to string and combine with Time column
    df['datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].str.split(' - ').str[0], format='%d-%m-%Y %H:%M', errors='coerce')
    df.set_index('datetime', inplace=True)
    df.drop(columns=['Day-ahead Total Load Forecast'], inplace=True)
    return df

# Function to add additional features
def add_features(df):
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    return df

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.xlsx'):
        file_path = os.path.join(folder_path, filename)
        
        # Load data from Excel workbook using openpyxl
        training_data = pd.read_excel(file_path, sheet_name='training_data', engine='openpyxl')
        validation_data = pd.read_excel(file_path, sheet_name='validation_data', engine='openpyxl')
        testing_data = pd.read_excel(file_path, sheet_name='testing_data', engine='openpyxl')
        
        # Trim any whitespace from the column names
        training_data.columns = training_data.columns.str.strip()
        validation_data.columns = validation_data.columns.str.strip()
        testing_data.columns = testing_data.columns.str.strip()
        
        # Check for NaN or infinite values
        print(f"Checking for NaN or infinite values in {filename}...")
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
        early_stopping_by_error = EarlyStoppingByError(start_epoch=1800)
        
        # Train the model with the callbacks
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=2000, batch_size=64, callbacks=[early_stopping, early_stopping_by_error])
        
        # Evaluate the model
        loss = model.evaluate(X_test, y_test)
        print(f'Test Loss for {filename}: {loss}')
        
        # Make predictions on the test data
        y_pred = model.predict(X_test)
        
        # Calculate error metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        print(f'Test Mean Squared Error for {filename}: {mse}')
        print(f'Test Mean Absolute Error for {filename}: {mae}')
        
        # Prepare the output DataFrame
        output_data = testing_data[['Date', 'Time']].copy()
        output_data['Predicted Load'] = y_pred
        
        # Ensure Date column is in the correct format
        output_data['Date'] = pd.to_datetime(output_data['Date'], format='%d-%m-%Y').dt.strftime('%d-%m-%Y')
        
        # Write the predictions to a new sheet in the same workbook, overwriting if it exists
        with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            output_data.to_excel(writer, sheet_name='future_predictions', index=False)
        
        print(f"Predictions for {filename} have been written to 'future_predictions' sheet.")