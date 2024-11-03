import numpy as np

def PREVENT_10yr_CVD_risk(sex, age, TC, HDL_C, TC_treat_status, SBP, BP_treat_status, if_diabetes, if_current_smoker, eGFR):
    """
    Calculate the PREVENT 10-year cardiovascular disease (CVD) risk based on various clinical indicators.
    
    Parameters
    ----------
    sex : int
        Patient's sex, where 0 represents female and 1 represents male.
        
    age : float
        Patient's age in years.
        
    TC : float
        Total cholesterol (TC) in mmol/L. If the input is in mg/dL, divide by 38.665 to convert to mmol/L.
        
    HDL_C : float
        High-density lipoprotein cholesterol (HDL-C) in mmol/L. If the input is in mg/dL, divide by 38.665 to convert to mmol/L.
        
    TC_treat_status : int
        Indicator of whether the patient is on treatment for total cholesterol, where 1 indicates on treatment, 0 indicates no treatment.
        
    SBP : float
        Systolic blood pressure (SBP) in mmHg.
        
    BP_treat_status : int
        Indicator of whether the patient is on blood pressure treatment, where 1 indicates on treatment, 0 indicates no treatment.
        
    if_diabetes : int
        Indicator of diabetes status, where 1 indicates the patient has diabetes, 0 indicates no diabetes.
        
    if_current_smoker : int
        Indicator of smoking status, where 1 indicates the patient is a current smoker, 0 indicates non-smoker.
        
    eGFR : float
        Estimated glomerular filtration rate (eGFR) in mL/min/1.73m², which is a measure of kidney function.

    Returns
    -------
    log_odds : float
        The calculated log odds of 10-year CVD risk based on the input clinical indicators.
        
    risk : float
        The calculated 10-year CVD risk probability (converted from log odds).
        
    Notes
    -----
    This function applies different coefficients for calculating log odds based on sex. The formula is derived 
    from the PREVENT model for assessing 10-year CVD risk. The model takes into account age adjustments 
    (e.g., `age_55_per_10_yr`) and various clinical indicators to produce a risk estimate.
    
    Examples
    --------
    >>> PREVENT_10yr_CVD_risk(sex=0, age=60, TC=200/38.665, HDL_C=50/38.665, TC_treat_status=1, 
                              SBP=130, BP_treat_status=1, if_diabetes=0, if_current_smoker=1, eGFR=75)
    (-2.1584405412166667, 0.10354511645091431)

    >>> risk = PREVENT_10yr_CVD_risk(sex=0, age=60, TC=6, HDL_C=1.7, TC_treat_status=1, 
                             SBP=130, BP_treat_status=1, if_diabetes=0, if_current_smoker=1, eGFR=75)[1]
    >>> print(f'10-year CVD risk: {risk*100:.2f}%')
    10-year CVD risk: 9.03%

    reference
    --------
    [1] Khan SS, Coresh J, Pencina MJ, et al. Novel Prediction Equations for Absolute Risk Assessment of Total Cardiovascular 
        Disease Incorporating Cardiovascular-Kidney-Metabolic Health: A Scientific Statement From the American Heart Association. 
        Circulation. 2023;148(24):1982-2004. doi:10.1161/CIR.0000000000001191

    [2] Khan SS, Matsushita K, Sang Y, et al. Development and Validation of the American Heart Association's PREVENT Equations. 
        Circulation. 2024;149(6):430-449. doi:10.1161/CIRCULATIONAHA.123.067626
    """
    # Transformations for each risk factor
    TC *= 38.665 # convert to mg/dL
    HDL_C *= 38.665 # convert to mg/dL
    
    age_55_per_10_yr = (age - 55) / 10
    non_HDL_C = TC - HDL_C
    cTC = (non_HDL_C * 0.02586 - 3.5) 
    cHDL = (HDL_C * 0.02586 - 1.3) / 0.3 
    cSBP_min = (min(SBP, 110) - 110) / 20
    cSBP_max = (max(SBP, 110) - 130) / 20
    ceGFR_min = (min(eGFR, 60) - 60) / -15
    ceGFR_max = (max(eGFR, 60) - 90) / -15

    # Coefficients for males and females
    if sex == 0:
        coefficients = {
            "intercept": -3.307728,
            "age": 0.7939329,
            "TC": 0.0305239,
            "HDL": -0.1606857,
            "SBP_min": -0.2394003,
            "SBP_max": 0.360078,
            "diabetes": 0.8667604,
            "smoking": 0.5360739,
            "eGFR_min": 0.6045917,
            "eGFR_max": 0.0433769,
            "BP_treat": 0.3151672,
            "TC_treat": -0.1477655,
            "BP_treat_SBP_max": -0.0663612,
            "TC_treat_TC": 0.1197879,
            "age_TC": -0.0819715,
            "age_HDL": 0.0306769,
            "age_SBP_max": -0.0946348,
            "age_diabetes": -0.27057,
            "age_smoking": -0.078715,
            "age_eGFR_min": -0.1637806
        }
    else:
        coefficients = {
            "intercept": -3.031168,
            "age": 0.7688528,
            "TC": 0.0736174,
            "HDL": -0.0954431,
            "SBP_min": -0.4347345,
            "SBP_max": 0.3362658,
            "diabetes": 0.7692857,
            "smoking": 0.4386871,
            "eGFR_min": 0.5378979,
            "eGFR_max": 0.0164827,
            "BP_treat": 0.288879,
            "TC_treat": -0.1337349,
            "BP_treat_SBP_max": -0.0475924,
            "TC_treat_TC": 0.150273,
            "age_TC": -0.0517874,
            "age_HDL": 0.0191169,
            "age_SBP_max": -0.1049477,
            "age_diabetes": -0.2251948,
            "age_smoking": -0.0895067,
            "age_eGFR_min": -0.1543702
        }

    # Calculate log odds
    log_odds = (
        coefficients["intercept"] +
        coefficients["age"] * age_55_per_10_yr +
        coefficients["TC"] * cTC +
        coefficients["HDL"] * cHDL +
        coefficients["SBP_min"] * cSBP_min +
        coefficients["SBP_max"] * cSBP_max +
        coefficients["diabetes"] * if_diabetes +
        coefficients["smoking"] * if_current_smoker +
        coefficients["eGFR_min"] * ceGFR_min +
        coefficients["eGFR_max"] * ceGFR_max +
        coefficients["BP_treat"] * BP_treat_status +
        coefficients["TC_treat"] * TC_treat_status +
        coefficients["BP_treat_SBP_max"] * BP_treat_status * cSBP_max +
        coefficients["TC_treat_TC"] * TC_treat_status * cTC +
        coefficients["age_TC"] * age_55_per_10_yr * cTC +
        coefficients["age_HDL"] * age_55_per_10_yr * cHDL +
        coefficients["age_SBP_max"] * age_55_per_10_yr * cSBP_max +
        coefficients["age_diabetes"] * age_55_per_10_yr * if_diabetes +
        coefficients["age_smoking"] * age_55_per_10_yr * if_current_smoker +
        coefficients["age_eGFR_min"] * age_55_per_10_yr * ceGFR_min
    )

    # Calculate 10-year risk
    risk = np.exp(log_odds) / (1 + np.exp(log_odds))
    
    return log_odds, risk


def SCORE2_10yr_CVD_risk(sex, age, SBP, TC, HDL_C, smoking_status, risk_region='low'):
    '''
    Calculate the 10-year cardiovascular disease (CVD) risk using the SCORE2 model for non-diabetic patients.

    Parameters
    ----------
    sex : int
        Patient's sex, where 0 represents female and 1 represents male.
        
    age : float
        Patient's age in years.
        
    SBP : float
        Systolic blood pressure (SBP) in mmHg.
        
    TC : float
        Total cholesterol (TC) in mmol/L.
        
    HDL_C : float
        High-density lipoprotein cholesterol (HDL-C) in mmol/L.
        
    smoking_status : int
        Smoking status, where 1 indicates the patient is a current smoker, 0 indicates non-smoker.
        
    risk_region : str, optional
        Risk region used for calibration. Possible values are "low", "moderate", "high", "very high", or None for uncalibrated risk.
    
    Returns
    -------
    x : float
        The calculated 10-year CVD risk log-odds.
        
    uncalibrated_risk : float, optional
        The uncalibrated 10-year CVD risk as a percentage. Only returned if `risk_region` is None.
        
    calibrated_risk : float, optional
        The calibrated 10-year CVD risk as a percentage. Only returned if `risk_region` is not None.

    Examples
    --------
    >>> risk = SCORE2_10yr_CVD_risk(sex=0, age=50, SBP=120, TC=7, HDL_C=1.2, smoking_status=0, risk_region='very high')[1]
    >>> print(f'10-year CVD risk: {risk*100:.2f}%')
    10-year CVD risk: 5.88%

    reference
    ----------
    [1] SCORE2 working group and ESC Cardiovascular risk collaboration. SCORE2 risk prediction algorithms: new models to 
        estimate 10-year risk of cardiovascular disease in Europe. Eur Heart J. 2021;42(25):2439-2454. doi:10.1093/eurheartj/ehab309

    [2] Visseren FLJ, Mach F, Smulders YM, et al. 2021 ESC Guidelines on cardiovascular disease prevention in clinical practice. 
        Eur Heart J. 2021;42(34):3227-3337. doi:10.1093/eurheartj/ehab484
    '''
    
    # Transformations for each risk factor
    cage = (age - 60) / 5
    csbp = (SBP - 120) / 20
    ctchol = TC - 6
    chdl = (HDL_C - 1.3) / 0.5
    
    # Coefficients for males and females
    if sex == 1:
        coefficients = {
            "age": 0.3742,
            "smoking": 0.6012,
            "sbp": 0.2777,
            "tchol": 0.1458,
            "hdl": -0.2698,
            "smoking_age": -0.0755,
            "sbp_age": -0.0255,
            "tchol_age": -0.0281,
            "hdl_age": 0.0426
        }
        baseline_survival = 0.9605
    else:
        coefficients = {
            "age": 0.4648,
            "smoking": 0.7744,
            "sbp": 0.3131,
            "tchol": 0.1002,
            "hdl": -0.2606,
            "smoking_age": -0.1088,
            "sbp_age": -0.0277,
            "tchol_age": -0.0226,
            "hdl_age": 0.0613
        }
        baseline_survival = 0.9776

    # Calculate x (sum of beta * transformed variables)
    x = (
        coefficients["age"] * cage +
        coefficients["smoking"] * smoking_status +
        coefficients["sbp"] * csbp +
        coefficients["tchol"] * ctchol +
        coefficients["hdl"] * chdl +
        coefficients["smoking_age"] * smoking_status * cage +
        coefficients["sbp_age"] * csbp * cage +
        coefficients["tchol_age"] * ctchol * cage +
        coefficients["hdl_age"] * chdl * cage
    )

    # Calculate uncalibrated 10-year risk
    uncalibrated_risk = 1 - (baseline_survival ** np.exp(x))

    if risk_region is None:
        return x, uncalibrated_risk
    else:
        # Risk calibration coefficients
        scales = {
            "low": (-0.5699, 0.7476, -0.7380, 0.7019),
            "moderate": (-0.1565, 0.8009, -0.3143, 0.7701),
            "high": (0.3207, 0.9360, 0.5710, 0.9369),
            "very high": (0.5836, 0.8294, 0.9412, 0.8329)
        }
        if risk_region not in scales:
            raise ValueError("Invalid risk_region. Choose from 'low', 'moderate', 'high', or 'very high'.")
        scale1, scale2 = scales[risk_region][:2] if sex == 1 else scales[risk_region][2:]

        # Calculate calibrated 10-year risk
        calibrated_risk = (1 - np.exp(-np.exp(scale1 + scale2 * np.log(-np.log(1 - uncalibrated_risk)))))
        return x, calibrated_risk


def SCORE2_diabetes_10yr_CVD_risk(sex, age, SBP, TC, HDL_C, smoking_status, a1c, eGFR, if_diabetes, age_diabetes=None, risk_region='low'):
    '''
    Calculate the 10-year cardiovascular disease (CVD) risk using the SCORE2-Diabetes model.
    
    Parameters
    ----------
    sex : int
        Patient's sex, where 0 represents female and 1 represents male.
        
    age : float
        Patient's age in years.
        
    SBP : float
        Systolic blood pressure (SBP) in mmHg.
        
    TC : float
        Total cholesterol (TC) in mmol/L. If the input is in mg/dL, divide by 38.665 to convert to mmol/L.
        
    HDL_C : float
        High-density lipoprotein cholesterol (HDL-C) in mmol/L. If the input is in mg/dL, divide by 38.665 to convert to mmol/L.
        
    if_diabetes : int
        Indicator of diabetes status, where 1 indicates the patient has diabetes, 0 indicates no diabetes.
        
    smoking_status : int
        Smoking status, where 1 indicates the patient is a current smoker, 0 indicates non-smoker.
        
    a1c : float
        HbA1c value in mmol/mol.
        
    eGFR : float
        Estimated glomerular filtration rate (eGFR) in mL/min/1.73m², which is a measure of kidney function.
        
    age_diabetes : float, optional
        Age at diabetes diagnosis. Only relevant if the patient has diabetes.
        
    risk_region : str, optional
        Risk region used for calibration. Possible values are "low", "moderate", "high", "very high", or None for uncalibrated risk.
    
    Returns
    -------
    x : float
        The calculated 10-year CVD risk log-odds.
        
    calibrated_risk : float, optional
        The calibrated 10-year CVD risk as a percentage. Only returned if `risk_region` is not None.

    Examples
    --------
    >>> SCORE2_diabetes_10yr_CVD_risk(sex=0, age=60,
                              SBP=160, TC=11, HDL_C=2,smoking_status=1,a1c=20, eGFR=90,
                              if_diabetes=1, age_diabetes=40, 
                              risk_region=None)
    (2.1496134657018633, 0.1766814956010121)

    >>> SCORE2_diabetes_10yr_CVD_risk(sex=1, age=50,
                              SBP=140, TC=6, HDL_C=1.7, smoking_status=0, a1c=30, eGFR=115,
                              if_diabetes=1,age_diabetes=40, 
                              risk_region='low')
    (-0.07974919731617436, 0.04715127192069235)

    reference
    ----------
    [1] SCORE2-Diabetes Working Group and the ESC Cardiovascular Risk Collaboration. SCORE2-Diabetes: 10-year cardiovascular risk 
        estimation in type 2 diabetes in Europe. Eur Heart J. 2023;44(28):2544-2556. doi:10.1093/eurheartj/ehad260

    [2] Visseren FLJ, Mach F, Smulders YM, et al. 2021 ESC Guidelines on cardiovascular disease prevention in clinical practice. 
        Eur Heart J. 2021;42(34):3227-3337. doi:10.1093/eurheartj/ehab484
    '''
    # Transformations for each risk factor
    cage = (age - 60) / 5
    csbp = (SBP - 120) / 20
    ctchol = TC - 6
    chdl = (HDL_C - 1.3) / 0.5
    ca1c = (a1c - 31) / 9.34
    cegfr = (np.log(eGFR) - 4.5) / 0.15
    cegfr2 = cegfr ** 2
    cagediab = if_diabetes * (age_diabetes - 50) / 5 if if_diabetes and age_diabetes else 0

    # Coefficients for males and females
    if sex == 1:
        coefficients = {
            "age": 0.5368,
            "smoking": 0.4774,
            "sbp": 0.1322,
            "diabetes": 0.6457,
            "tchol": 0.1102,
            "hdl": -0.1087,
            "smoking_age": -0.0672,
            "sbp_age": -0.0268,
            "diabetes_age": -0.0983,
            "tchol_age": -0.0181,
            "hdl_age": 0.0095,
            "a1c": 0.0955,
            "egfr": -0.0591,
            "egfr2": 0.0058,
            "a1c_age": -0.0134,
            "egfr_age": 0.0115,
            "cagediab": -0.0998
        }
        baseline_survival = 0.9605
    else:
        coefficients = {
            "age": 0.6624,
            "smoking": 0.6139,
            "sbp": 0.1421,
            "diabetes": 0.8096,
            "tchol": 0.1127,
            "hdl": -0.1568,
            "smoking_age": -0.1122,
            "sbp_age": -0.0167,
            "diabetes_age": -0.1272,
            "tchol_age": -0.0200,
            "hdl_age": 0.0186,
            "a1c": 0.1173,
            "egfr": -0.0640,
            "egfr2": 0.0062,
            "a1c_age": -0.0196,
            "egfr_age": 0.0169,
            "cagediab": -0.1180
        }
        baseline_survival = 0.9776

    # Calculate x (sum of beta * transformed variables)
    x = (
        coefficients["age"] * cage +
        coefficients["smoking"] * smoking_status +
        coefficients["sbp"] * csbp +
        coefficients["diabetes"] * if_diabetes +
        coefficients["tchol"] * ctchol +
        coefficients["hdl"] * chdl +
        coefficients["smoking_age"] * smoking_status * cage +
        coefficients["sbp_age"] * csbp * cage +
        coefficients["diabetes_age"] * if_diabetes * cage +
        coefficients["tchol_age"] * ctchol * cage +
        coefficients["hdl_age"] * chdl * cage +
        coefficients["a1c"] * ca1c +
        coefficients["egfr"] * cegfr +
        coefficients["egfr2"] * cegfr2 +
        coefficients["a1c_age"] * ca1c * cage +
        coefficients["egfr_age"] * cegfr * cage +
        coefficients["cagediab"] * cagediab
    )

    # Calculate uncalibrated 10-year risk
    risk = 1 - (baseline_survival ** np.exp(x))

    if risk_region is None:
        return x, risk
    else:
        # Risk calibration coefficients
        scales = {
            "low": (-0.5699, 0.7476, -0.7380, 0.7019),
            "moderate": (-0.1565, 0.8009, -0.3143, 0.7701),
            "high": (0.3207, 0.9360, 0.5710, 0.9369),
            "very high": (0.5836, 0.8294, 0.9412, 0.8329)
        }
        if risk_region not in scales:
            raise ValueError("Invalid risk_region. Choose from 'low', 'moderate', 'high', or 'very high'.")
        scale1, scale2 = scales[risk_region][:2] if sex == 1 else scales[risk_region][2:]

        # Calculate calibrated 10-year risk
        calibrated_risk = (1 - np.exp(-np.exp(scale1 + scale2 * np.log(-np.log(1 - risk)))))
        return x, calibrated_risk


def PCE_10yr_CVD_risk(sex, age, Chol, HDL_C, SBP, BP_treat_status, if_current_smoker, if_diabetes):
    """
    Calculate the Pooled Cohort Equations (PCE) 10-year cardiovascular disease (CVD) risk.
    
    Parameters
    ----------
    sex : int
        Patient's sex, where 0 represents female and 1 represents male.
        
    age : float
        Patient's age in years at baseline.
        
    Chol : float
        Total cholesterol in mmol/L. If the input is in mg/dL, divide by 38.665 to convert.
        
    HDL_C : float
        High-density lipoprotein cholesterol (HDL-C) in mmol/L. If the input is in mg/dL, divide by 38.665 to convert.
        
    SBP : float
        Systolic blood pressure (SBP) in mmHg.
        
    BP_treat_status : int
        Indicates if the patient is on blood pressure treatment, 1 for yes and 0 for no.
        
    if_current_smoker : int
        Smoking status, where 1 indicates the patient is a current smoker, 0 indicate a non-smoker.
        
    if_diabetes : int
        Indicator of diabetes status, where 1 indicates the patient has diabetes, 0 indicates no diabetes.

    Returns
    -------
    log_odds : float
        The calculated log odds of 10-year CVD risk.
        
    risk : float
        The calculated 10-year CVD risk probability.

    Examples
    -------
    >>> PCE_10yr_CVD_risk(sex=0, age=60, Chol=200/38.665, HDL_C=60/38.665, SBP=145, BP_treat_status=1, if_current_smoker=1, if_diabetes=0)
    (-27.94619365741534, 0.11043213358879855)

    >>> PCE_10yr_CVD_risk(sex=0, age=60, Chol=11, HDL_C=2, SBP=145, BP_treat_status=1, if_current_smoker=1, if_diabetes=0)
    (-27.523756427168653, 0.16350417037705245)

    reference
    --------
    [1] Andrus B, Lacaille D. 2013 ACC/AHA guideline on the assessment of cardiovascular risk. 
        J Am Coll Cardiol. 2014;63(25 Pt A):2886. doi:10.1016/j.jacc.2014.02.606

    [2] Karmali KN, Goff DC Jr, Ning H, Lloyd-Jones DM. A systematic examination of the 2013 ACC/AHA pooled cohort risk assessment tool for 
        atherosclerotic cardiovascular disease. J Am Coll Cardiol. 2014;64(10):959-968. doi:10.1016/j.jacc.2014.06.1186

    [3] Rana JS, Tabada GH, Solomon MD, et al. Accuracy of the Atherosclerotic Cardiovascular Risk Equation in a Large Contemporary, Multiethnic Population. 
        J Am Coll Cardiol. 2016;67(18):2118-2130. doi:10.1016/j.jacc.2016.02.055
    """
    # Transformations for each risk factor
    log_age = np.log(age)
    log_age_squared = log_age ** 2
    TC_mg_dL = Chol * 38.665  # convert mmol/L to mg/dL
    log_TC = np.log(TC_mg_dL)
    HDL_mg_dL = HDL_C * 38.665  # convert mmol/L to mg/dL
    log_hdl = np.log(HDL_mg_dL)
    log_SBP = np.log(SBP)
    current_smoker = if_current_smoker
    diabetes = if_diabetes

    # Interaction terms
    log_age_x_log_TC = log_age * log_TC
    log_age_x_log_hdl = log_age * log_hdl
    log_age_x_current_smoker = log_age * current_smoker

    # Coefficients and baseline for each sex
    if sex == 0:  # Female
        coefficients = {
            "intercept": -29.18,
            "baseline_surv": 0.9665,
            "log_age": -29.799,
            "log_age_squared": 4.884,
            "log_TC": 13.540,
            "log_age_x_log_TC": -3.114,
            "log_hdl": -13.578,
            "log_age_x_log_hdl": 3.149,
            "log_SBP_treated": 2.019,
            "log_SBP_untreated": 1.957,
            "current_smoker": 7.574,
            "log_age_x_current_smoker": -1.665,
            "diabetes": 0.661
        }
    else:  # Male
        coefficients = {
            "intercept": 61.18,
            "baseline_surv": 0.9144,
            "log_age": 12.344,
            "log_TC": 11.853,
            "log_age_x_log_TC": -2.664,
            "log_hdl": -7.990,
            "log_age_x_log_hdl": 1.769,
            "log_SBP_treated": 1.797,
            "log_SBP_untreated": 1.764,
            "current_smoker": 7.837,
            "log_age_x_current_smoker": -1.795,
            "diabetes": 0.658
        }

    # Selecting the correct coefficient for SBP treatment status
    sbp_coef = coefficients["log_SBP_treated"] if BP_treat_status == 1 else coefficients["log_SBP_untreated"]

    # Calculating log odds
    log_odds = (
        coefficients["log_age"] * log_age +
        (coefficients.get("log_age_squared", 0) * log_age_squared if sex == 0 else 0) +
        coefficients["log_TC"] * log_TC +
        coefficients["log_age_x_log_TC"] * log_age_x_log_TC +
        coefficients["log_hdl"] * log_hdl +
        coefficients["log_age_x_log_hdl"] * log_age_x_log_hdl +
        sbp_coef * log_SBP +
        coefficients["current_smoker"] * current_smoker +
        coefficients["log_age_x_current_smoker"] * log_age_x_current_smoker +
        coefficients["diabetes"] * diabetes
    )

    # Calculate 10-year risk
    mean_risk = coefficients["intercept"]
    baseline_surv = coefficients["baseline_surv"]
    risk = 1 - baseline_surv ** np.exp(log_odds - mean_risk)
    
    return log_odds, risk