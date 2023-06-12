import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression


# read student data
data = pd.read_csv("student_data.csv")

# create input title
title = [
    "freetime", "goout", "Dalc", "Walc", "activities", "famsize", "Pstatus", "famrel", "health"
]

# create data_x without helath and data_y with health from data title 
data_x = data[title[:-1]]
data_y = data[title[-1]]

# Create data_x one hot encoder
ohe = OneHotEncoder(sparse_output=False)

# Fit and transform data_x
encoder = ohe.fit_transform(data_x.iloc[:, 4:7])

# Concat the encoder and data_x from 4 to 7 with iloc
data_x = pd.concat([data_x.iloc[:, :4], pd.DataFrame(encoder), data_x.iloc[:, 7::]], axis=1)

# Split data to train and test
x_train, x_test, y_train, y_test = train_test_split(data_x.values, data_y.values, test_size=0.2, random_state=2, shuffle=True)


# Create LogisticRegression model
model = LogisticRegression()

# Fit model
model.fit(x_train, y_train)

# Predict model
y_pred = model.predict(x_test)

# Print score
print("Score: ", model.score(x_test, y_test))

# Print coef
print("Coef: ", model.coef_)

# Print intercept
print("Intercept: ", model.intercept_)

# Print predict line one by one
for i in range(len(y_pred)):
    print("Predict: ", y_pred[i], "Actual: ", y_test[i])




filename = 'life_balance.sav'
pickle.dump(model, open(filename, 'wb'))