import pandas as pd

# Load data from Excel workbook
file_path = 'machine_learning/data/model.xlsx'
training_data = pd.read_excel(file_path, sheet_name='training_data')
validation_data = pd.read_excel(file_path, sheet_name='validation_data')
testing_data = pd.read_excel(file_path, sheet_name='testing_data')

# Combine date and time into a single datetime column
def preprocess_data(df):
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    df.set_index('datetime', inplace=True)
    df.drop(columns=['date', 'time', 'predicted_load'], inplace=True)
    return df

training_data = preprocess_data(training_data)
validation_data = preprocess_data(validation_data)
testing_data = preprocess_data(testing_data)

# Display the first few rows of the training data
print(training_data.head())

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Prepare the data for training
X_train = training_data.index.values.reshape(-1, 1)
y_train = training_data['actual_load'].values

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Validate the model
X_val = validation_data.index.values.reshape(-1, 1)
y_val = validation_data['actual_load'].values
y_val_pred = model.predict(X_val)

# Calculate validation error
val_error = mean_squared_error(y_val, y_val_pred)
print(f'Validation Mean Squared Error: {val_error}')

# Test the model
X_test = testing_data.index.values.reshape(-1, 1)
y_test = testing_data['actual_load'].values
y_test_pred = model.predict(X_test)

# Calculate test error
test_error = mean_squared_error(y_test, y_test_pred)
print(f'Test Mean Squared Error: {test_error}')