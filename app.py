from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)

CAUSES = [
    "Labour",
    "Material",
    "Equipment",
    "Management",
    "Environment",
    "Design / Scope",
]

DEFAULT_AHP = np.array([
    [1,   1/2, 2,   2,   3,   2],
    [2,   1,   2,   2,   3,   2],
    [1/2, 1/2, 1,   1,   2,   1],
    [1/2, 1/2, 1,   1,   2,   1],
    [1/3, 1/3, 1/2, 1/2, 1,   1/2],
    [1/2, 1/2, 1,   1,   2,   1],
], dtype=float)

# Illustrative ML coefficients fallback so the dashboard works immediately.
# Run ml/train_model.py to generate the real Random Forest model.
def ahp_weights(matrix):
    matrix = np.array(matrix, dtype=float)
    values, vectors = np.linalg.eig(matrix)
    idx = np.argmax(values.real)
    lam = float(values[idx].real)
    weights = np.abs(vectors[:, idx].real)
    weights = weights / weights.sum()

    n = len(CAUSES)
    ci = (lam - n) / (n - 1)
    ri = {3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24}[n]
    cr = ci / ri
    return weights, lam, ci, cr

def severity(days, planned):
    if days <= 0:
        return 0
    p = days / planned * 100
    if p < 2: return 1
    if p < 5: return 2
    if p < 10: return 3
    if p < 20: return 4
    return 5

def ml_fallback(features):
    # Transparent fallback for first launch before model training.
    s = np.array(features[:6], dtype=float)
    planned = float(features[6])
    return max(0.0, float(s @ np.array([2.5, 2.8, 2.3, 2.7, 1.8, 3.0]) + planned * 0.025 - 12))

@app.route("/")
def index():
    return render_template("index.html", causes=CAUSES, default_ahp=DEFAULT_AHP.tolist())

@app.post("/api/analyze")
def analyze():
    data = request.get_json(force=True)
    planned = float(data.get("planned_days", 0))

    if planned <= 0:
        return jsonify({"error": "Planned project duration must be greater than zero."}), 400

    delay_map = data.get("delays", {})
    days = np.array([max(0.0, float(delay_map.get(c, 0))) for c in CAUSES])

    matrix = np.array(data.get("ahp_matrix", DEFAULT_AHP.tolist()), dtype=float)
    weights, lam, ci, cr = ahp_weights(matrix)

    weighted = float(np.sum(days * weights))
    severities = [severity(d, planned) for d in days]

    # Use trained Random Forest if available.
    ml_delay = None
    try:
        from ml.predict import predict_delay
        ml_delay = predict_delay({
            "labour_severity": severities[0],
            "material_severity": severities[1],
            "equipment_severity": severities[2],
            "management_severity": severities[3],
            "environment_severity": severities[4],
            "design_scope_severity": severities[5],
            "planned_days": planned,
        })
        ml_source = "Trained Random Forest model"
    except Exception:
        ml_delay = ml_fallback(severities + [planned])
        ml_source = "Transparent fallback; run ml/train_model.py"

    # AHP-only contribution scaled to total-factor space for the prototype.
    ahp_component = weighted * len(CAUSES)
    hybrid = 0.50 * ahp_component + 0.50 * ml_delay
    hybrid = min(hybrid, planned * 2)

    rows = []
    for i, cause in enumerate(CAUSES):
        rows.append({
            "cause": cause,
            "days": round(float(days[i]), 2),
            "ahp_weight": round(float(weights[i]), 4),
            "weighted_days": round(float(days[i] * weights[i]), 2),
            "severity": severities[i],
        })

    return jsonify({
        "entered_delay_days": round(float(days.sum()), 2),
        "ahp_weighted_delay": round(ahp_component, 2),
        "ml_forecast_delay": round(float(ml_delay), 2),
        "hybrid_forecast_delay": round(float(hybrid), 2),
        "forecast_completion_days": round(planned + hybrid, 2),
        "delay_percent": round(hybrid / planned * 100, 2),
        "lambda_max": round(lam, 4),
        "ci": round(ci, 4),
        "cr": round(cr, 4),
        "consistency_status": "Acceptable" if cr <= 0.10 else "Review comparisons",
        "ml_source": ml_source,
        "causes": rows,
    })

if __name__ == "__main__":
    app.run(debug=True)
