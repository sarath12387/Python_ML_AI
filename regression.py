import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
path = r"C:\Users\pottu\Downloads\Salary_Data.csv"
df = pd.read_csv(path)

# Check the data
print(df.head())
print("Shape:", df.shape)
print("Columns:", df.columns)
print("Nulls:\n", df.isnull().sum())
print("Data types:\n", df.dtypes)

# Define features and target
X = df.drop('Salary', axis=1) # YearsExperience
y = df['Salary']

# Split the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

# Model training
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X_train, y_train)

# Make predictions
y_predictions = LR.predict(X_test)

# Evaluation metrics
from sklearn.metrics import r2_score, mean_squared_error
R2 = r2_score(y_test, y_predictions)
MSE = mean_squared_error(y_test, y_predictions)
RMSE = np.sqrt(MSE)

print("\nEvaluation:")
print("R-square:", R2)
print("MSE:", MSE)
print("RMSE:", RMSE)

# Manual MSE calculation
manual_mse = np.mean((y_test.values - y_predictions) ** 2)
print("Manual MSE:", manual_mse)

# Coefficients
print("\nThe coefficient of Years of Experience is:", LR.coef_)
print("Intercept is:", LR.intercept_)

# Use VarianceThreshold to check if features have zero variance
from sklearn.feature_selection import VarianceThreshold
vt = VarianceThreshold(threshold=0)
vt.fit(X)

# Correct: check selected features
print("Selected features mask:", vt.get_support())
print("Selected columns:", X.columns[vt.get_support()])

# Fix: Manually compute variance instead of vt.variances_
print("Manual feature variances:\n", X.var())

# Save model using pickle
import pickle
pickle.dump(LR, open('YearsExperience_model.pkl', 'wb'))

# Load the model and make a test prediction
model = pickle.load(open('YearsExperience_model.pkl', 'rb'))
ip1 = [[5]] # 5 years of experience
print("\nPrediction for 5 years experience:", model.predict(ip1))