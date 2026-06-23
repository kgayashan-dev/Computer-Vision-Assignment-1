# AI Assignment_1_CSC2602

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import drive

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

# Load dataset from Google Drive
drive.mount('/content/drive')

# housing.csv was downloaded from Kaggle and uploaded to Google Drive
DATA_PATH = "/content/drive/MyDrive/datasets/california_housing/housing.csv"
df = pd.read_csv(DATA_PATH)

# View first few rows
# display(df.head()) # can use for viewing

# Dataset information
df.info()



# Prepare X and y
df = df.dropna()

# Remove categorical column for this simple program
df = df.drop(columns=["ocean_proximity"])

X = df.drop(columns=["median_house_value"])
y = df["median_house_value"]


# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# Improve model performance by adjusting hyperparameters
param_grid = {

    "n_estimators": [50, 100],

    "max_depth": [5, 10],

    "min_samples_split": [2, 5],

    "min_samples_leaf": [1, 2]

}

model = RandomForestRegressor(

    random_state=42,

    n_jobs=-1

)

grid = GridSearchCV(

    model,

    param_grid,

    cv=2,

    scoring="r2",

    n_jobs=-1

)

grid.fit(X_train, y_train)
best_model = grid.best_estimator_

# Evaluation
y_pred = best_model.predict(X_test)

print("Best Params:", grid.best_params_)
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R^2 Score:", r2_score(y_test, y_pred)) # 82%

# Plotting predicted vs actual
plt.figure(figsize=(8, 6))

sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)

# Ideal prediction
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red",
    label="Ideal Fit (y = x)"
)

plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Prices")
plt.legend()
plt.grid(True)

plt.show()