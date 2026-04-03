from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load model + columns
model = pickle.load(open('model.pkl', 'rb'))
model_columns = pickle.load(open('columns.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Inputs
        temperature = float(request.form.get('temperature'))
        humidity = float(request.form.get('humidity'))
        wind_speed = float(request.form.get('wind_speed'))
        pressure = float(request.form.get('average_pressure'))
        visibility = float(request.form.get('visibility'))

        # 🔥 Feature Engineering (same as training)
        temp_humidity = temperature * humidity
        wind_pressure = wind_speed * pressure
        temp_squared = temperature ** 2
        humidity_squared = humidity ** 2
        wind_squared = wind_speed ** 2

        # Create input dictionary
        input_data = {
            'temperature': temperature,
            'humidity': humidity,
            'wind_speed': wind_speed,
            'average_pressure': pressure,
            'visibility': visibility,
            'temp_humidity': temp_humidity,
            'wind_pressure': wind_pressure,
            'temp_squared': temp_squared,
            'humidity_squared': humidity_squared,
            'wind_squared': wind_squared
        }

        # Convert to DataFrame
        df = pd.DataFrame([input_data])

        # Match training columns
        df = df.reindex(columns=model_columns, fill_value=0)

        # Predict
        prediction = model.predict(df)
        output = round(prediction[0], 2)

        return render_template(
            'index.html',
            prediction_text=f"Predicted Power Output: {output}"
        )

    except Exception as e:
        print("Error:", e)
        return render_template(
            'index.html',
            prediction_text="Error in input values"
        )

if __name__ == "__main__":
    app.run(debug=True)