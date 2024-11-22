# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Splitting data into training and testing sets
from sklearn.model_selection import train_test_split

# Predicting the probability of a binary outcome using the logistic function
from sklearn.linear_model import LogisticRegression

# Measuring the proportion of correctly classified instances in a classification model
from sklearn.metrics import accuracy_score



# Loading dataset
data = pd.read_csv("diabetes.csv")

# Displaying data
data

# Checking for missing values
sns.heatmap(data.isnull())


# Train-test split
X = data.drop("Outcome", axis=1)
Y = data['Outcome']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Training the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# Making predictions
prediction = model.predict(X_test)

# Displaying predictions
print(prediction)

# Calculating accuracy
accuracy = accuracy_score(prediction, Y_test)

# Displaying accuracy
print(accuracy)

# Save the trained model to a file
joblib.dump(model, 'trained_model.joblib')
