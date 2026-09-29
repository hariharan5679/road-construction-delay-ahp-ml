# Hybrid AHP + Machine Learning for Road Construction Delay Analysis

A research-oriented web application for analysing and forecasting project delays in road construction using **Analytic Hierarchy Process (AHP)** and **Machine Learning (ML)**.

## Research title

**A Hybrid Approach for Delay Analysis and Forecasting in Road Construction Projects Using AHP and Machine Learning**

## Project objective

The system accepts observed delay information from a road construction project and combines:

1. **AHP** — converts expert judgement into relative importance weights for major delay factors.
2. **Machine Learning** — learns the relationship between delay-factor severity and historical project delay.
3. **Hybrid forecasting** — combines the AHP-based delay contribution and ML prediction to estimate overall project delay.

> **Academic note:** The included dataset is synthetic/illustrative and is intended only for demonstrating the workflow. Do not present it as field data or use it to claim validated real-world predictive accuracy.

## Delay factors

- Labour
- Material
- Equipment
- Management
- Environment
- Design / Scope

## System architecture

```text
User Input
   |
   +--> Delay days by factor
   |
   +--> AHP pairwise comparison matrix
             |
             v
        AHP calculation
        - Eigenvector weights
        - CI
        - CR
             |
             +-------------------+
                                 |
Delay severity + project data --> ML model
                                 |
                                 v
                         ML delay forecast
                                 |
             AHP result ---------+
                    |
                    v
             Hybrid forecast
                    |
                    v
       Overall project delay
       + forecast completion
```

## Main features

- Browser-based Flask dashboard
- Planned project duration input
- Cause-wise delay-day input
- Editable AHP pairwise comparison matrix
- AHP weights and consistency ratio
- ML prediction
- Hybrid AHP + ML forecast
- Cause-wise contribution table
- Sample CSV dataset
- Reproducible ML training script
- Saved model support
- Academic methodology documentation

## Folder structure

```text
road-construction-delay-ahp-ml/
│
├── app.py
├── requirements.txt
├── .gitignore
├── .env.example
├── README.md
│
├── data/
│   └── road_delay_sample.csv
│
├── ml/
│   ├── train_model.py
│   └── predict.py
│
├── models/
│   └── .gitkeep
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
└── docs/
    ├── methodology.md
    └── research_notes.md
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/road-construction-delay-ahp-ml.git
cd road-construction-delay-ahp-ml
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Train the ML model

```bash
python ml/train_model.py
```

The training script reads:

```text
data/road_delay_sample.csv
```

and saves the trained model under:

```text
models/
```

The generated model is ignored by Git so that large/binary model files do not unnecessarily enter version control.

## Run the application

```bash
python app.py
```

Open the local Flask URL shown in the terminal, normally:

```text
http://127.0.0.1:5000
```

## ML methodology

The sample implementation uses a **Random Forest Regressor**.

Input variables include:

- Labour severity
- Material severity
- Equipment severity
- Management severity
- Environment severity
- Design/scope severity
- Planned project duration

Target:

- Actual delay days

The training script reports:

- MAE
- RMSE
- R²

For an academic study, these metrics should be calculated using a sufficiently large real-world dataset and preferably cross-validation.

## AHP methodology

The AHP module uses the principal eigenvector method.

For a pairwise comparison matrix:

```text
A × w = λmax × w
```

where:

- `A` = pairwise comparison matrix
- `w` = priority vector
- `λmax` = maximum eigenvalue

Consistency Index:

```text
CI = (λmax - n) / (n - 1)
```

Consistency Ratio:

```text
CR = CI / RI
```

A commonly used research threshold is:

```text
CR <= 0.10
```

A value above this threshold indicates that the expert comparisons should be reviewed.

## Hybrid model

The prototype uses:

```text
Hybrid Forecast =
0.50 × AHP-based delay
+
0.50 × ML forecast
```

The 50:50 weighting is a **prototype assumption**, not a universal scientific rule.

For the dissertation, test different hybrid weights on validation data and select/calibrate the combination using an objective validation criterion.

## Recommended real project dataset

Replace the sample data with project records containing, where available:

- Project ID
- Planned duration
- Actual duration
- Actual delay days
- Labour delay/severity
- Material delay/severity
- Equipment delay/severity
- Management delay/severity
- Environmental delay/severity
- Design/scope delay/severity
- Contract type
- Road type
- Project length
- Project value
- Weather/environment variables
- Number of variation orders
- Approval/inspection delays

## Academic validation

For a dissertation-quality study:

1. Collect historical road project data.
2. Clean missing and inconsistent values.
3. Define delay-factor variables consistently.
4. Obtain expert pairwise comparisons for AHP.
5. Calculate AHP weights and CR.
6. Train ML models using historical project records.
7. Compare multiple ML algorithms.
8. Use train/test split or cross-validation.
9. Calculate MAE, RMSE and R².
10. Calibrate the AHP/ML hybrid weighting.
11. Test the final model on unseen projects.
12. Compare predicted delay with observed delay.
13. Discuss limitations and generalisability.

## Important research limitation

A prediction model is only as credible as its training data and validation design. The sample CSV included here is synthetic and should be replaced by real project observations before reporting model accuracy or drawing conclusions.

## License

This repository is intended as an academic/research project. Add an institutional or open-source license if required by your university or supervisor.
