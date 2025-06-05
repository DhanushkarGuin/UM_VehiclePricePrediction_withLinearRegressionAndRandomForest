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

unique_values = dataset['cylinders'].unique()
print(f"Unique values: {unique_values}")
print(f"Count: {len(unique_values)}")

# Random Forest is performing better in this case!
# Will use Random Forest for User Input Prediction

make = input('Enter the manufacturer of the vehicle:')
model = input('Enter the model name of the vehicle:')
year = float(input('Enter the year of manufacturing:'))
engine = input('Enter the engine, including type and specifications:')
cylinders = float(input('Enter number of cylinders in the vehicle:'))
fuel = input('Enter the type of fuel:')
mileage = float(input('Enter the mileage in miles:'))
transmission = input('Enter the type of transmission:')
trim = input('Enter the trim level of the vehicle, indicating different feature sets or packages:')
body = input('Enter the body style of the vehicle:')
doors = float(input('Enter the number of doors:'))
exterior_color = input('Enter the exterior color of vehicle:')
interior_color = input('Enter the exterior color of vehicle:')
drivetrain = input('Enter the drivetrain of vehicle:')

make_encoded = [0] * 28
make_index = {
    'jeep': 0, 'gmc': 1, 'dodge': 2, 'ram': 3, 'nissan': 4, 'ford': 5,
    'hyundai': 6, 'chevrolet': 7, 'volkswagen': 8, 'chrysler': 9, 'kia': 10,
    'mazda': 11, 'acura': 12, 'subaru': 13, 'audi': 14, 'bmw': 15, 'toyota': 16,
    'buick': 17, 'mercedes-benz': 18, 'honda': 19, 'lincoln': 20, 'cadillac': 21,
    'infiniti': 22, 'lexus': 23, 'land rover': 24, 'volvo': 25, 'genesis': 26,
    'jaguar': 27
}
if make.lower() in make_index:
    make_encoded[make_index[make.lower()]] = 1


all_model = [
    'Wagoneer', 'Grand Cherokee', 'Yukon XL', 'Durango', '3500', 'Murano', 'F-350',
    'Tucson Hybrid', 'Compass', 'Santa Cruz', 'Blazer EV', 'Explorer', 'Taos',
    'Jetta', 'Hornet', 'Tucson', 'Terrain', 'Pacifica', '2500', 'Transit-250',
    'Grand Cherokee 4xe', 'EV6', 'Mustang Mach-E', 'Silverado 1500', 'Seltos',
    'Blazer', 'Wrangler', 'CX-90 PHEV', 'MDX', 'Outback', 'Atlas Cross Sport',
    'IONIQ 5', 'Q8 e-tron', 'Sonata Hybrid', 'EV9', 'Sportage Hybrid',
    'Expedition', 'Atlas', 'Grand Cherokee L', 'X7', 'Tundra Hybrid', 'Kicks',
    'Envista', 'Rogue', 'Wagoneer L', 'Sportage', 'EQE 350+', 'Sonata', 'Ranger',
    'SQ5', 'Q5 e', 'Charger', 'Sprinter 2500', 'CR-V Hybrid', 'IONIQ 6',
    'Transit-150', 'Santa Fe', 'Equinox', 'Solterra', 'Wrangler 4xe',
    'ProMaster 2500', 'Corsair', 'Savana 2500', 'Pacifica Hybrid', 'ZDX', 'Altima',
    'EQS 450', 'ProMaster 3500', 'XT5', 'Gladiator', 'QX50', 'Versa', 'AMG GLE 53',
    'A5 Sportback', 'Nautilus', 'Trailblazer', 'i4 Gran Coupe', 'Escape',
    'Bronco Sport', 'Elantra HEV', 'RX 500h', 'Prologue', 'i5', 'MX-5 Miata RF',
    'ID.4', 'CR-V', 'Silverado 2500', 'Sierra 3500', 'Legacy', '300', 'i7',
    'M235 Gran Coupe', 'Range Rover Evoque', 'Odyssey', 'Sierra 2500', 'Titan',
    'RS e-tron GT', 'Edge', 'Discovery Sport', 'Sprinter 3500',
    'XC90 Recharge Plug-In Hybrid', 'Sorento', 'Frontier', 'HR-V', 'Niro',
    'Encore GX', 'Electrified G80', 'GLE 450', 'GLS 450', 'Trax', 'Defender',
    'SQ8 e-tron', 'Sorento Hybrid', 'Envision', 'Bronco', 'Palisade', '740',
    'Passport', 'LYRIQ', 'Corolla Cross', 'Equinox EV', 'Grand Wagoneer', 'Kona',
    'QX55', 'Corolla', 'F-150', 'A3', 'GLE 350', 'Enclave', 'Sentra', 'I-PACE',
    'RAV4 Prime', 'Voyager', 'Sierra 1500', 'Transit Connect', 'Maverick',
    'Transit-350', 'Pathfinder', '530', 'GLA 250', 'Telluride', 'CX-90',
    'Electrified GV70', 'S60 Recharge Plug-In Hybrid', 'XT6', 'Impreza', 'CX-70',
    'X3', 'C40 Recharge Pure Electric', 'ProMaster 1500', 'Soul', 'AMG C 43',
    'Forte'
]

model_index = {model: idx for idx, model in enumerate(all_model)}
model_encoded = [0] * 153
if model.lower() in model_index:
    model_encoded[model_index[model.lower()]] = 1


year_index = {2024: 0, 2023: 1,2025: 2}
year_encoded = [0,0,0]
if year in year_index:
    year_encoded[year_index[year]] = 1

all_engine = [
    '24V GDI DOHC Twin Turbo', 'OHV',
    '6.2L V-8 gasoline direct injection, variable valve contr', '16V MPFI OHV',
    '24V DDI OHV Turbo Diesel', '24V MPFI DOHC', '32V DDI OHV Turbo Diesel',
    '16V GDI DOHC Turbo Hybrid',
    'ar 3.6L V-6 DOHC, variable valve control, regular unleade',
    '16V GDI DOHC Turbo', 'oled Turbo Diesel I-6 6.7 L/408',
    '16V PDI DOHC Turbo', 'c',
    '4 gasoline direct injection, DOHC, variable valve control',
    '16V PDI DOHC', 'DOHC', '24V PDI DOHC Flexible Fuel', '16V GDI OHV',
    '16V MPFI DOHC', '24V GDI DOHC',
    'gasoline direct injection, DOHC, intercooled turbo, premi',
    '16V GDI DOHC Hybrid', '24V GDI SOHC',
    'gasoline direct injection, DOHC, variable valve control,', 'c Motor',
    '4 gasoline direct injection, DOHC, Multiair variable valv',
    '4 gasoline direct injection, DOHC, CVVT variable valve co',
    '24V PDI DOHC Twin Turbo',
    'ream 2.5L I-4 port/direct injection, DOHC, CVVT variable',
    '24V GDI DOHC Turbo',
    'MAX 3.4L V-6 port/direct injection, DOHC, variable valve',
    '4 DOHC, CVTCS variable valve control, regular unleaded, e',
    '12V GDI DOHC Turbo',
    'o 1.5L I-3 gasoline direct injection, DOHC, CVTCS variabl',
    'ne 3L I-6 gasoline direct injection, DOHC, variable valve',
    '4 port/direct injection, DOHC, CVVT variable valve contro',
    'oled Turbo Premium Unleaded V-6 3.0 L/183',
    'High Output 6.7L I-6 diesel direct injection, VVT interc',
    'DOHC 16V LEV3-SULEV30', '16V DDI DOHC Turbo Diesel', 'Turbo DOHC',
    'TIV-G 2.5L I-4 gasoline direct injection, DOHC, variable',
    '6 gasoline direct injection, variable valve control, regu',
    '24V MPFI DOHC Hybrid', '16V GDI DOHC',
    'ne 2L I-4 gasoline direct injection, DOHC, variable valve',
    'o 2L I-4 port/direct injection, DOHC, variable valve cont',
    '8 variable valve control, regular unleaded, engine with 4',
    '1.3L I-3 gasoline direct injection, DOHC, variable valve',
    't 1.5L I-3 port/direct injection, DOHC, Ti-VCT variable v', 'oBoost',
    '>\n\n    \n    <dt>VIN</dt>\n     ZACNDFAN0R3A12168', '24V VVT',
    '.6L I-4 gasoline direct injection, DOHC, D-CVVT variable',
    '6.7L I-6 diesel direct injection, VVT intercooled turbo,',
    '16V PDI DOHC Turbo Hybrid', '24V DDI DOHC Turbo Diesel', ';',
    'dd>\n\n    \n    <dt>VIN</dt>\n     1V2BMPE85RC003636',
    'gasoline direct injection, DOHC, i-VTEC variable valve co',
    '6.6L V-8 diesel direct injection, intercooled turbo, die', 'der',
    '12V PDI DOHC Turbo', 'oled Turbo Premium Unleaded I-4 2.0 L/122',
    'ce 5.6L V-8 gasoline direct injection, DOHC, variable val', 'c ZEV',
    '4 port/direct injection, DOHC, variable valve control, in',
    'Unleaded V-8 5.6 L/339', '12V GDI OHV',
    't 2L I-4 gasoline direct injection, DOHC, variable valve',
    '6 port/direct injection, DOHC, variable valve control, re',
    '6 gasoline direct injection, i-VTEC variable valve contro',
    '6 DOHC, variable valve control, regular unleaded, engine',
    '5.3L V-8 gasoline direct injection, variable valve contr',
    '6 DOHC, VVT variable valve control, engine with cylinder',
    'ar 3.6L V-6 DOHC, VVT variable valve control, regular unl',
    '32V GDI DOHC Twin Turbo', '24V DOHC',
    '<dt>VIN</dt>\n     3GN7DNRPXRS232327',
    'diesel direct injection, DOHC, intercooled turbo, diesel,', 'Unknown',
    '24V PDI DOHC Twin Turbo Hybrid', 'EcoBoost',
    '<dt>VIN</dt>\n     1FMUK7HH1SGA05728', 'ER',
    '<dt>VIN</dt>\n     3C63R3HLXRG198198',
    'ream 1.6L I-4 gasoline direct injection, DOHC, CVVD varia',
    'd>\n\n    \n    <dt>VIN</dt>\n     SADHM2S12R1631756',
    '16V PDI DOHC Hybrid', 'c ZEV 320hp',
    'gasoline direct injection, DOHC, iVCT variable valve cont',
    '4 DOHC, variable valve control, regular unleaded, engine',
    '6 gasoline direct injection, DOHC, variable valve control',
    'rbo Regular Unleaded I-6 3.0 L/183',
    'II 3.8L V-6 gasoline direct injection, DOHC, D-CVVT varia',
    'd>\n\n    \n    <dt>VIN</dt>\n     7FARS4H71SE000866',
    '16V MPFI DOHC Hybrid', 'o',
    '4L V-8 VVT variable valve control, regular unleaded, engi',
    'DOHC, D-CVVT variable valve control, regular unleaded, en',
    '8 gasoline direct injection, variable valve control, regu'
]

engine_index = {engine: idx for idx, engine in enumerate(all_engine)}
engine_encoded = [0] * 153
if engine.lower() in engine_index:
    engine_encoded[engine_index[engine.lower()]] = 1

cylinder_index = {6.: 0, 8.: 1, 4.: 2, 4.9754738: 3, 3.: 4, 0.: 5}
cylinder_encoded = [0,0,0,0,0,0]
if cylinders in cylinder_index:
    cylinder_encoded[cylinder_index[cylinders]] = 1