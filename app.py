import pandas as pd

# save filepath to variable for easier access
melbourne_file_path = "./melb_data.csv"

# read the data and store data in DataFrame titled melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path)

# print a summary of the data in Melbourne data
print(melbourne_data.describe())

# Your First Machine Learning Model
print(melbourne_data.columns)

# dropna drops missing values (think of na as "not available")
melbourne_data = melbourne_data.dropna(axis=0)

# Selecting The Prediction Target
y = melbourne_data.Price

# Choosing "Features"
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]
print(" --- Describe X --- ")
print(X.describe())

print(" --- Head of X --- ")
print(X.head())

# Building Your Model
from sklearn.tree import DecisionTreeRegressor

# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)

# Fit model
melbourne_model.fit(X, y)

# Make predictions for the first 5 houses. These are the predicted prices.
print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))


