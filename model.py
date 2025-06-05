import numpy as np
import pandas as pd

dataset = pd.read_csv('dataset.csv')
# print(dataset.head())

dataset.drop(['name'], axis= 1, inplace=True)
# print(dataset.head())

## I did clean the description data but still the one hot encoded names would have been an issue with such long descriptions so i just dropped it
dataset.drop(['description'], axis=1,inplace=True)
# print(dataset.head())

dataset.fillna({'make': 'Unknown'}, inplace=True)
dataset.fillna({'model': 'Unknown'}, inplace=True)
dataset.fillna({'engine': 'Unknown'}, inplace=True)
dataset.fillna({'fuel': 'Unknown'}, inplace=True)
dataset.fillna({'transmission': 'Unknown'}, inplace=True)
dataset.fillna({'trim': 'Unknown'}, inplace=True)
dataset.fillna({'body': 'Unknown'}, inplace=True)
dataset.fillna({'exterior_color': 'Unknown'}, inplace=True)
dataset.fillna({'interior_color': 'Unknown'}, inplace=True)
dataset.fillna({'drivetrain': 'Unknown'}, inplace=True)

from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=3)  
dataset[['year']] = imputer.fit_transform(dataset[['year']])
dataset[['price']] = imputer.fit_transform(dataset[['price']])
dataset[['cylinders']] = imputer.fit_transform(dataset[['cylinders']])
dataset[['mileage']] = imputer.fit_transform(dataset[['mileage']])
dataset[['doors']] = imputer.fit_transform(dataset[['doors']])

X = dataset.iloc[:, [i for i in range(dataset.shape[1]) if i != 3]]
y = dataset.iloc[:, 3]

# print(X.columns)

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(sparse_output=False), [0,1,3,5,7,8,9,10,11,12,13])], 
                    remainder='passthrough')
X = np.array(ct.fit_transform(X))

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

df_lr = pd.DataFrame({'Real Values': y_test, 'Predicted Values': y_pred})
# print(df_lr)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train_scaled, y_train)

y_rf_pred = rf.predict(X_test_scaled)

df_rf = pd.DataFrame({'Real Values': y_test, 'Predicted Values': y_rf_pred})
# print(df_rf)

from sklearn.metrics import mean_squared_error,r2_score

mse_lr = mean_squared_error(y_test,y_pred)
mse_rf = mean_squared_error(y_test,y_rf_pred)

rmse_lr = np.sqrt(mse_lr)  
r2_lr = r2_score(y_test, y_pred)

rmse_rf = np.sqrt(mse_rf)  
r2_rf = r2_score(y_test, y_rf_pred)

print('RMSE for LR:', rmse_lr)
print('r2 for LR:', r2_lr)

print('RMSE for rf:', rmse_rf)
print('r2 for rf:', r2_rf)

# Random Forest is performing better in this case!