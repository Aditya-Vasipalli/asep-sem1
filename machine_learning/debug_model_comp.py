import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# Define the file paths
model_file_path = 'machine_learning/data/model.xlsx'
comparison_file_path = 'machine_learning/data/comparison.xlsx'

# Check if the files exist
if not os.path.exists(model_file_path):
    raise FileNotFoundError(f"The file {model_file_path} does not exist. Please check the path.")
if not os.path.exists(comparison_file_path):
    raise FileNotFoundError(f"The file {comparison_file_path} does not exist. Please check the path.")

# Load data from Excel workbooks using openpyxl
training_data = pd.read_excel(model_file_path, sheet_name='training_data', engine='openpyxl')
validation_data = pd.read_excel(model_file_path, sheet_name='validation_data', engine='openpyxl')
testing_data = pd.read_excel(model_file_path, sheet_name='testing_data', engine='openpyxl')
comparison_data = pd.read_excel(comparison_file_path, sheet_name='comparison_data', engine='openpyxl')

# Commented out logging with log nametag
# print("[LOG] Training Data Columns:", training_data.columns)
# print("[LOG] Validation Data Columns:", validation_data.columns)
# print("[LOG] Testing Data Columns:", testing_data.columns)

# Trim any whitespace from the column names
training_data.columns = training_data.columns.str.strip()
validation_data.columns = validation_data.columns.str.strip()
testing_data.columns = testing_data.columns.str.strip()
comparison_data.columns = comparison_data.columns.str.strip()

# Combine date and time into a single datetime column
def preprocess_data(df):
    # Extract the start time from the time range
    df['Time'] = df['Time'].str.split(' - ').str[0]
    # Convert Date to string and combine with Time, then parse to datetime
    df['datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'], errors='coerce')
    df.set_index('datetime', inplace=True)
    df.drop(columns=['Date', 'Time', 'Day-ahead Total Load Forecast'], inplace=True)
    return df

training_data = preprocess_data(training_data)
validation_data = preprocess_data(validation_data)
testing_data = preprocess_data(testing_data)
comparison_data = preprocess_data(comparison_data)

# Check for and handle missing values
training_data = training_data.dropna(subset=['Actual Total Load'])
validation_data = validation_data.dropna(subset=['Actual Total Load'])
testing_data = testing_data.dropna(subset=['Actual Total Load'])
comparison_data = comparison_data.dropna(subset=['Actual Total Load'])

# Debugging: Print the contents of comparison_data
print("[DEBUG] Comparison Data after preprocessing and dropping NaNs:")
print(comparison_data)

# Add additional features
def add_features(df):
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    return df

training_data = add_features(training_data)
validation_data = add_features(validation_data)
testing_data = add_features(testing_data)
comparison_data = add_features(comparison_data)

# Debugging: Print the contents of comparison_data after adding features
print("[DEBUG] Comparison Data after adding features:")
print(comparison_data)

# Prepare the data for training
X_train = training_data[['hour', 'day_of_week', 'month']].values
y_train = training_data['Actual Total Load'].values

X_val = validation_data[['hour', 'day_of_week', 'month']].values
y_val = validation_data['Actual Total Load'].values

X_test = testing_data[['hour', 'day_of_week', 'month']].values
y_test = testing_data['Actual Total Load'].values

X_comparison = comparison_data[['hour', 'day_of_week', 'month']].values
y_comparison = comparison_data['Actual Total Load'].values

# Train the model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Validate the model
y_val_pred = model.predict(X_val)

# Calculate validation error
val_error = mean_squared_error(y_val, y_val_pred)
print(f'Validation Mean Squared Error: {val_error}')

# Test the model
y_test_pred = model.predict(X_test)

# Calculate test error
test_error = mean_squared_error(y_test, y_test_pred)
print(f'Test Mean Squared Error: {test_error}')

# Make predictions on comparison data
if len(X_comparison) > 0:
    y_comparison_pred = model.predict(X_comparison)

    # Calculate comparison error
    comparison_error = mean_squared_error(y_comparison, y_comparison_pred)
    print(f'Comparison Mean Squared Error: {comparison_error}')

    # Add predictions to comparison data
    comparison_data['Predicted Total Load'] = y_comparison_pred

    # Print comparison data with predictions
    print(comparison_data)
else:
    print("[ERROR] No data available for comparison.")