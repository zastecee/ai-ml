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

# Building Your Model



