import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor




title = [
    "freetime",
    "failures",
    "studytime",
    "absences",
    "internet",
    "higher",
    "schoolsup",
    "Dalc",
    "Walc",
    "G1",
    "G2",
    "G3"
]

data = pd.read_csv("student_data.csv")

# Create input title wihout G3
data_x = data[title[:-1]]


# Create output title with G3
data_y = data[title[-1]]

# Create data_x one hot encoder
ohe = OneHotEncoder(sparse_output=False)

# Fit and transform data_x
encoder = ohe.fit_transform(data_x.iloc[:, 4:7])

# Concat the encoder and data_x from 4 to 7 with iloc
data_x = pd.concat([data_x.iloc[:, :4], pd.DataFrame(encoder), data_x.iloc[:, 7::]], axis=1)

print(data_x)

# Split data to train and test
x_train, x_test, y_train, y_test = train_test_split(data_x.values, data_y.values, test_size=0.2, random_state=0, shuffle=True)

# Create model
model = GradientBoostingRegressor()

print("GradientBoostingRegressorScore")

# Fit model
model.fit(x_train, y_train)

# Score model
print(f"\nGradientBoostingRegressorScore: {model.score(x_test, y_test)}\n")

# Predict model
for index, item in enumerate(x_test):
    print(f"\tPredict: {int(model.predict([item])[0])} - Real: {y_test[index]}")
    
print("\n")

filename = 'academis_success.sav'
pickle.dump(model, open(filename, 'wb'))