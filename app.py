import os
import pickle
import pandas as pd
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "Random_Forest-Algorithm.pkl"
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# HTML Template with all 21 inputs
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Predictive Intelligence Portal</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0b0f19;
            --card-bg: #111827;
            --border-color: #1f2937;
            --text-primary: #f3f4f6;
            --text-secondary: #9ca3af;
            --accent-color: #6366f1;
            --accent-hover: #4f46e5;
            --shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
        body { background-color: var(--bg-color); color: var(--text-primary); min-height: 100vh; display: flex; flex-direction: column; align-items: center; padding: 40px 20px; }
        .container { width: 100%; max-width: 950px; background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 16px; box-shadow: var(--shadow); padding: 40px; margin-bottom: 30px; }
        h1 { font-size: 1.8rem; font-weight: 700; margin-bottom: 8px; background: linear-gradient(135deg, #818cf8 0%, #c084fc 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; }
        p.subtitle { text-align: center; color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 30px; }
        .form-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 15px; }
        .form-group { display: flex; flex-direction: column; }
        label { font-size: 0.8rem; font-weight: 500; color: var(--text-secondary); margin-bottom: 6px; }
        input { background: #0f172a; border: 1px solid var(--border-color); border-radius: 8px; padding: 10px 12px; color: var(--text-primary); font-size: 0.9rem; transition: all 0.3s ease; }
        input:focus { outline: none; border-color: var(--accent-color); box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2); }
        .btn-container { grid-column: 1 / -1; margin-top: 20px; }
        button { width: 100%; background: var(--accent-color); color: white; border: none; border-radius: 8px; padding: 14px; font-size: 1rem; font-weight: 600; cursor: pointer; transition: background 0.3s ease; box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4); }
        button:hover { background: var(--accent-hover); }
        .result-card { width: 100%; max-width: 950px; background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 16px; padding: 25px; text-align: center; margin-top: 20px; box-shadow: var(--shadow); }
        .result-card h3 { font-size: 1rem; margin-bottom: 8px; color: var(--text-secondary); text-transform: uppercase; }
        .result-value { font-size: 1.5rem; font-weight: 700; color: #34d399; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Model Inference Portal</h1>
        <p class="subtitle">Provide all 21 feature inputs required by the Random Forest model</p>
        
        <form method="POST">
            <div class="form-grid">
                <div class="form-group"><label>Company Name</label><input type="text" name="company_name" required placeholder="e.g. TechCorp"></div>
                <div class="form-group"><label>Industry</label><input type="text" name="industry" required placeholder="e.g. Software"></div>
                <div class="form-group"><label>Country</label><input type="text" name="country" required placeholder="e.g. USA"></div>
                <div class="form-group"><label>Company Size</label><input type="text" name="company_size" required placeholder="e.g. Large"></div>
                <div class="form-group"><label>Month</label><input type="number" name="month" required placeholder="e.g. 5"></div>
                <div class="form-group"><label>Year</label><input type="number" name="year" required placeholder="e.g. 2026"></div>
                <div class="form-group"><label>Layoffs Count</label><input type="number" name="layoffs_count" required placeholder="e.g. 150"></div>
                <div class="form-group"><label>Layoff Percentage (%)</label><input type="number" step="any" name="layoff_percentage" required placeholder="e.g. 12.5"></div>
                <div class="form-group"><label>Reason For Layoffs</label><input type="text" name="reason_for_layoffs" required placeholder="e.g. Cost cutting"></div>
                <div class="form-group"><label>AI Automation Impact</label><input type="number" step="any" name="ai_automation_impact" required placeholder="e.g. 7.5"></div>
                <div class="form-group"><label>AI Replacement Risk</label><input type="number" step="any" name="ai_replacement_risk" required placeholder="e.g. 6.0"></div>
                <div class="form-group"><label>Open Roles</label><input type="number" name="open_roles" required placeholder="e.g. 25"></div>
                <div class="form-group"><label>Hiring Trend</label><input type="text" name="hiring_trend" required placeholder="e.g. Declining"></div>
                <div class="form-group"><label>Remote Jobs Percentage (%)</label><input type="number" step="any" name="remote_jobs_percentage" required placeholder="e.g. 40.0"></div>
                <div class="form-group"><label>Top Hiring Role</label><input type="text" name="top_hiring_role" required placeholder="e.g. Engineer"></div>
                <div class="form-group"><label>Stock Growth Percent (%)</label><input type="number" step="any" name="stock_growth_percent" required placeholder="e.g. 5.2"></div>
                <div class="form-group"><label>Revenue Growth Percent (%)</label><input type="number" step="any" name="revenue_growth_percent" required placeholder="e.g. 8.1"></div>
                <div class="form-group"><label>Salary Budget Change (%)</label><input type="number" step="any" name="salary_budget_change" required placeholder="e.g. 2.0"></div>
                <div class="form-group"><label>AI Adoption Level</label><input type="text" name="ai_adoption_level" required placeholder="e.g. High"></div>
                <div class="form-group"><label>Employee Sentiment</label><input type="text" name="employee_sentiment" required placeholder="e.g. Neutral"></div>
                <div class="form-group"><label>Job Security Score</label><input type="number" step="any" name="job_security_score" required placeholder="e.g. 65.0"></div>
                
                <div class="btn-container">
                    <button type="submit">Predict Outcome</button>
                </div>
            </div>
        </form>
    </div>

    {% if prediction is not none %}
    <div class="result-card">
        <h3>Prediction Result</h3>
        <div class="result-value">{{ prediction }}</div>
    </div>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            data = {
                'company_name': [request.form['company_name']],
                'industry': [request.form['industry']],
                'country': [request.form['country']],
                'company_size': [request.form['company_size']],
                'month': [float(request.form['month'])],
                'year': [float(request.form['year'])],
                'layoffs_count': [float(request.form['layoffs_count'])],
                'layoff_percentage': [float(request.form['layoff_percentage'])],
                'reason_for_layoffs': [request.form['reason_for_layoffs']],
                'ai_automation_impact': [float(request.form['ai_automation_impact'])],
                'ai_replacement_risk': [float(request.form['ai_replacement_risk'])],
                'open_roles': [float(request.form['open_roles'])],
                'hiring_trend': [request.form['hiring_trend']],
                'remote_jobs_percentage': [float(request.form['remote_jobs_percentage'])],
                'top_hiring_role': [request.form['top_hiring_role']],
                'stock_growth_percent': [float(request.form['stock_growth_percent'])],
                'revenue_growth_percent': [float(request.form['revenue_growth_percent'])],
                'salary_budget_change': [float(request.form['salary_budget_change'])],
                'ai_adoption_level': [request.form['ai_adoption_level']],
                'employee_sentiment': [request.form['employee_sentiment']],
                'job_security_score': [float(request.form['job_security_score'])]
            }
            
            df = pd.DataFrame(data)
            
            if model is not None:
                pred = model.predict(df)
                prediction = pred[0]
            else:
                prediction = "Model failed to load."
        except Exception as e:
            prediction = f"Error: {e}"
            
    return render_template_string(HTML_TEMPLATE, prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)