import pandas as pd
from sklearn.linear_model import LinearRegression


df = pd.read_csv("Crop_recommendation.csv")


X = df[['temperature', 'humidity']]
y = df['rainfall']


model = LinearRegression()
model.fit(X, y)


new_temp_value = float(input("Enter temperature (tempavg): "))
new_humidity_value = float(input("Enter humidity (humidity avg): "))

new_data = pd.DataFrame({'temperature': [new_temp_value], 'humidity': [new_humidity_value]})

predicted_rainfall = model.predict(new_data)

print("Predicted Rainfall:", predicted_rainfall[0])
