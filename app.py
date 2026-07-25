import pickle
import numpy as np
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Path to your trained model file
MODEL_PATH = 'Decisin_Regressor_model (1).pkl'

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    model = None
    print(f"Error loading model file '{MODEL_PATH}': {e}")

# HTML Template with modern styling & responsive UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Insurance Cost Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #4f46e5;
            --primary-hover: #4338ca;
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
            --card-bg: rgba(255, 255, 255, 0.96);
            --text-main: #0f172a;
            --text-muted: #64748b;
            --border-color: #e2e8f0;
            --accent-green: #059669;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            min-height: 100vh;
            background: var(--bg-gradient);
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 2rem 1rem;
        }

        .container {
            width: 100%;
            max-width: 680px;
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            border-radius: 20px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.35);
            padding: 2.5rem;
            animation: fadeIn 0.6s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(15px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .header {
            text-align: center;
            margin-bottom: 2rem;
        }

        .header h1 {
            font-size: 1.85rem;
            font-weight: 700;
            color: var(--text-main);
            letter-spacing: -0.02em;
            margin-bottom: 0.5rem;
        }

        .header p {
            color: var(--text-muted);
            font-size: 0.95rem;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.25rem;
        }

        @media (max-width: 600px) {
            .form-grid {
                grid-template-columns: 1fr;
            }
        }

        .form-group {
            display: flex;
            flex-direction: column;
        }

        label {
            font-size: 0.875rem;
            font-weight: 600;
            color: #334155;
            margin-bottom: 0.4rem;
        }

        input, select {
            width: 100%;
            padding: 0.75rem 1rem;
            font-size: 0.95rem;
            border: 1.5px solid var(--border-color);
            border-radius: 10px;
            background-color: #f8fafc;
            color: var(--text-main);
            transition: all 0.2s ease;
            outline: none;
        }

        input:focus, select:focus {
            border-color: var(--primary);
            background-color: #ffffff;
            box-shadow: 0 0 0 4px rgba(79, 70, 229, 0.15);
        }

        .btn-submit {
            margin-top: 1.5rem;
            width: 100%;
            padding: 0.9rem;
            font-size: 1rem;
            font-weight: 600;
            color: #ffffff;
            background-color: var(--primary);
            border: none;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.2s ease;
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
        }

        .btn-submit:hover {
            background-color: var(--primary-hover);
            transform: translateY(-1px);
            box-shadow: 0 6px 16px rgba(79, 70, 229, 0.4);
        }

        .result-card {
            margin-top: 2rem;
            padding: 1.5rem;
            border-radius: 12px;
            background: #f0fdf4;
            border: 1px solid #bbf7d0;
            text-align: center;
            animation: slideUp 0.4s ease-out;
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .result-card h3 {
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #166534;
            margin-bottom: 0.3rem;
        }

        .result-card .price {
            font-size: 2.25rem;
            font-weight: 800;
            color: var(--accent-green);
        }

        .error-card {
            margin-top: 2rem;
            padding: 1rem 1.25rem;
            border-radius: 10px;
            background: #fef2f2;
            border: 1px solid #fecaca;
            color: #991b1b;
            font-size: 0.9rem;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Insurance Cost Predictor</h1>
            <p>Enter individual details below to predict medical insurance cost</p>
        </div>

        <form action="/predict" method="POST">
            <div class="form-grid">
                <div class="form-group">
                    <label for="age">Age</label>
                    <input type="number" id="age" name="age" min="1" max="120" placeholder="e.g. 28" required value="{{ request.form.get('age', '') }}">
                </div>

                <div class="form-group">
                    <label for="sex">Gender</label>
                    <select id="sex" name="sex" required>
                        <option value="" disabled selected>Select Gender</option>
                        <option value="0" {% if request.form.get('sex') == '0' %}selected{% endif %}>Female</option>
                        <option value="1" {% if request.form.get('sex') == '1' %}selected{% endif %}>Male</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="bmi">BMI (Body Mass Index)</label>
                    <input type="number" step="0.1" id="bmi" name="bmi" min="10" max="60" placeholder="e.g. 26.5" required value="{{ request.form.get('bmi', '') }}">
                </div>

                <div class="form-group">
                    <label for="children">Number of Children</label>
                    <input type="number" id="children" name="children" min="0" max="10" placeholder="e.g. 0" required value="{{ request.form.get('children', '') }}">
                </div>

                <div class="form-group">
                    <label for="smoker">Smoker Status</label>
                    <select id="smoker" name="smoker" required>
                        <option value="" disabled selected>Select Status</option>
                        <option value="0" {% if request.form.get('smoker') == '0' %}selected{% endif %}>Non-Smoker</option>
                        <option value="1" {% if request.form.get('smoker') == '1' %}selected{% endif %}>Smoker</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="region">Region</label>
                    <select id="region" name="region" required>
                        <option value="" disabled selected>Select Region</option>
                        <option value="0" {% if request.form.get('region') == '0' %}selected{% endif %}>Northeast</option>
                        <option value="1" {% if request.form.get('region') == '1' %}selected{% endif %}>Northwest</option>
                        <option value="2" {% if request.form.get('region') == '2' %}selected{% endif %}>Southeast</option>
                        <option value="3" {% if request.form.get('region') == '3' %}selected{% endif %}>Southwest</option>
                    </select>
                </div>
            </div>

            <button type="submit" class="btn-submit">Predict Insurance Cost</button>
        </form>

        {% if prediction_text %}
        <div class="result-card">
            <h3>Estimated Medical Insurance Cost</h3>
            <div class="price">${{ prediction_text }}</div>
        </div>
        {% endif %}

        {% if error_text %}
        <div class="error-card">
            {{ error_text }}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template_string(
            HTML_TEMPLATE, 
            error_text=f"Model file '{MODEL_PATH}' could not be loaded. Ensure the file is present in the project directory."
        )

    try:
        # Extract features from form input
        age = float(request.form['age'])
        sex = float(request.form['sex'])
        bmi = float(request.form['bmi'])
        children = float(request.form['children'])
        smoker = float(request.form['smoker'])
        region = float(request.form['region'])

        # Prepare feature array: ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
        features = np.array([[age, sex, bmi, children, smoker, region]])

        # Predict using loaded model
        prediction = model.predict(features)[0]
        output = f"{prediction:,.2f}"

        return render_template_string(HTML_TEMPLATE, prediction_text=output)

    except Exception as e:
        return render_template_string(HTML_TEMPLATE, error_text=f"Error in prediction: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
