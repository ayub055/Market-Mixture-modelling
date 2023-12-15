from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae
import pandas as pd
import tqdm as tqdm

# read data

data = pd.read_csv('data/Advertising.csv')
data = data.copy().drop(['Unnamed: 0'], axis = 1)
X = data.loc[:, data.columns != 'sales']
y = data.loc[:, ['sales']]
split = int(len(X) * 0.8)
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

print("\n------------Traininf Model ------------")
# instantiate model
# model = RandomForestRegressor()
model = LinearRegression(positive=True)
model.fit(X_train, y_train.values.ravel())
pred = model.predict(X_test)
# print(pred)
print(pred.shape)
print(y_test.shape)
# print(f"Accuracy : {acc}")

mae_val = mae(y_test, pred)
print(f"MAE : {mae_val}")

# Extracting Feature Importance

print("\n------------Calculating FE------------")
ft = pd.Series(model.feature_importances_, index = X_train.columns)
print(ft)
# ft.nlargest(25).plot(kind='barh', figsize=(10,10))





# print(y_test, pred)