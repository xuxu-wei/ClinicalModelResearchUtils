# ClinicalModelResearchUtils
## Introduction
In our research journey, we noticed a lack of suitable Python implementations for commonly used clinical prediction models, formulas, and evaluation metrics. To address this gap, we created this repository and are sharing our work to support researchers who are passionate about using Python for clinical prediction model research.

It is not designed for any specific research project. As our research evolves, this repository will continue to be updated and expanded.

We hope this repository is helpful for you as well! 🚀

## Features
1. Cardiovascular Risk Prediction Models
PREVENT 10-Year CVD Risk: Calculates the 10-year CVD risk using the PREVENT model, incorporating clinical indicators such as age, cholesterol levels, blood pressure, and kidney function.
SCORE2 10-Year CVD Risk: Estimates 10-year CVD risk for non-diabetic patients based on the European SCORE2 framework.
SCORE2-Diabetes: Extends SCORE2 to include diabetes-specific variables like HbA1c and age at diabetes diagnosis.
Pooled Cohort Equations (PCE): Implements the widely used PCE model for 10-year CVD risk assessment.
2. Utility Functions
eGFR Calculation: Computes estimated glomerular filtration rate (eGFR) using the 2021 CKD-EPI formula, optionally integrating cystatin C.
HbA1c Conversion: Converts HbA1c values between DCCT (%) and IFCC (mmol/mol) units.
3. Model Evaluation Metrics
C-Index Validation: Ensures data validity for survival analysis and calculates concordance indices.
Reclassification Metrics: Includes continuous and categorical net reclassification improvement (NRI) tools for model comparison.
IDI Calculation: Computes integrated discrimination improvement (IDI) to quantify the added predictive value of new models.
4. Data Resampling
Bootstrap Resampling: Provides a resampling generator for robust statistical evaluation, supporting multiple datasets and iterations.

## Usage
Here’s a quick example of calculating CVD risk using the PREVENT model:
```python
from guideline_models import PREVENT_10yr_CVD_risk

log_odds, risk = PREVENT_10yr_CVD_risk(
    sex=0, age=60, TC=6, HDL_C=1.7, TC_treat_status=1,
    SBP=130, BP_treat_status=1, if_diabetes=0,
    if_current_smoker=1, eGFR=75
)
print(f"10-year CVD Risk: {risk*100:.2f}%")
```

For model evaluation:
```python
from metrix import get_continuous_nri

nri_event, nri_nonevent, continuous_nri = get_continuous_nri(
    y_true, old_model_preds, new_model_preds
)
print(f"Continuous NRI: {continuous_nri:.4f}")
```

## Contact
For issues, suggestions, or collaborations, please create an issue in this repository or reach out to the repository owner.


