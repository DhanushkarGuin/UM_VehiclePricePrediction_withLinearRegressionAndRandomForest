import pickle
import pandas as pd

model = pickle.load(open('model.pkl', 'rb'))

columns = ['make', 'model', 'year',
            'engine', 'cylinders', 
            'fuel', 'mileage', 'transmission', 
            'trim', 'body', 'doors', 'exterior_color', 
            'interior_color', 'drivetrain']

test_input = pd.DataFrame([['Hyundai', 'Tucson Hybrid', 2024, '16V GDI DOHC Turbo Hyrbid',
                            4,'Hybrid',5,'6-Speed Automatic',
                            'Limited','SUV',4,'White Pearl',
                            'Black', 'All Wheel Drive'
                            ]], columns=columns)

prediction = model.predict(test_input)
print(f'Predicted Price:', prediction)