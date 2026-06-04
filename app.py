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
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
                        'YearBuilt', 'Lattitude', 'Longtitude']
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

# What is Model Validation
from sklearn.metrics import mean_absolute_error

predicted_home_prices = melbourne_model.predict(X)
print("--- Mean Absolute Error ---")
print(mean_absolute_error(y, predicted_home_prices))

# The Problem with "In-Sample" Scores
# The measure we just computed can be called an "in-sample" score. We used a single "sample" of houses for both building the model and evaluating it.

# Out-of-sample
from sklearn.model_selection import train_test_split
# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Define model
melbourne_model_2 = DecisionTreeRegressor()

# Fit model
melbourne_model_2.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = melbourne_model_2.predict(val_X)
print("--- Mean Absolute Error In Validation Data ---")
print(mean_absolute_error(val_y, val_predictions))


# print the top few validation predictions
print(val_predictions[:5])
# print the top few actual prices from validation data
print(val_y[:5])

results = pd.DataFrame({
    "Actual": val_y.values,
    "Predicted": val_predictions
})
print(results.head())

# results = pd.DataFrame({
#     "Actual": val_y.values,
#     "Predicted": val_predictions
# })
# results.head()

# print("First in-sample predictions:", iowa_model.predict(X.head()))
# print("Actual target values for those homes:", y.head().tolist())
