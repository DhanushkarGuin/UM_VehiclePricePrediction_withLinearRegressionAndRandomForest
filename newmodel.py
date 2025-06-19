import pandas as pd

dataset = pd.read_csv('dataset.csv')
# print(dataset.head())

dataset.drop(['name'], axis= 1, inplace=True)
## I did clean the description data but still the one hot encoded names would have been an issue with such long descriptions so i just dropped it
dataset.drop(['description'], axis=1,inplace=True)
# print(dataset.head())

# print(dataset.isnull().sum())
# price - numerical - mean
# cylinders - numerical - mode
# fuel - categorical - mode
# mileage - numerical - mean
# transmission - categorical - mode
# trim - categorical - mode
# body - categorical - mode
# doors - numerical - mode
# exterior_color - categorical - mode
# interior_color- categorical - mode

dataset['price'] = dataset['price'].fillna(dataset['price'].mean())

X = dataset.drop(columns=['price'])
y = dataset['price']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

numerical_features = ['year','cylinders', 'mileage', 'doors']
categorical_features = ['make','model','engine','fuel', 'transmission', 'trim', 'body', 'exterior_color', 'interior_color', 'drivetrain']

numerical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numerical_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features)
])

from sklearn.ensemble import RandomForestRegressor
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import root_mean_squared_error, r2_score
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Root Mean Squared Error: {rmse}')
print(f'R2 Score: {r2}')

# print(dataset.columns.tolist())

import pickle
pickle.dump(model, open('model.pkl', 'wb'))