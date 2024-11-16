import numpy as np

def eGFR_2021_CKD_EPI(sex, age, S_cr, S_cys=None):
    '''
    Calculate the estimated Glomerular Filtration Rate (eGFR) using the 2021 CKD-EPI equation, 
    which takes into account serum creatinine (S_cr) and optionally serum cystatin C (S_cys).
    
    Parameters
    ----------
    sex : int
        Binary variable indicating biological sex.
        - 0 : Female
        - 1 : Male
    
    age : float
        Age of the individual in years. 
    
    S_cr : float
        Serum Creatinine in mg/dL, required for all calculations. 
    
    S_cys : float, optional
        Serum Cystatin C in mg/L. If provided, the improved 2021 CKD-EPI equation will be used.
    
    Returns
    -------
    float
        Estimated Glomerular Filtration Rate (eGFR) in ml/min/1.73 m².
    
    Notes
    -----
    The function applies two different equations depending on whether serum cystatin C (S_cys) 
    is provided:
    
    - Base formula (without S_cys):
        This calculation is based solely on serum creatinine (S_cr) and is less precise than 
        the improved formula but remains a standard for eGFR estimation.
    
    - Improved formula (with S_cys):
        Uses both serum creatinine (S_cr) and serum cystatin C (S_cys), providing a more 
        accurate estimation of eGFR, especially valuable in cases where S_cys data is available.
    
    reference
    ----------
    [1] Inker LA, Eneanya ND, Coresh J, et al. New Creatinine- and Cystatin C-Based Equations to Estimate GFR without Race. 
    N Engl J Med. 2021;385(19):1737-1749. doi:10.1056/NEJMoa2102953
    '''
    assert ((sex == 0) or (sex == 1)), f'sex code must be 0 (female) or 1 (male)'

    k = {0: 0.7, 1: 0.9}[sex]
    
    # 2021 CKD-EPI Creatinine
    if S_cys is None:
        miu = 142
        a1 = {0: -0.329, 1: -0.411}[sex]
        a2 = -1.209
        b1 = 0
        b2 = 0
        c = 0.9938
        d = {0: 1.012, 1: 1}[sex] # *d if female
        S_cys = 0
        
    # 2021 CKD-EPI Creatinine-Cystatin C
    else:
        miu = 135
        a1 = {0: -0.219, 1: -0.144}[sex]
        a2 = -0.544
        b1 = -0.323
        b2 = -0.778
        c = 0.9961
        d = {0: 0.963, 1: 1}[sex]
        
    eGFR = miu * min(S_cr/k, 1)**a1 * max(S_cr/k, 1)**a2 * min(S_cys/0.8, 1)**b1 * max(S_cys/0.8, 1)**b2 * c**age * d
    return eGFR


def dcct_to_ifcc(dcct_hba1c):
    """
    Convert HbA1c from the DCCT (%) unit to the IFCC (mmol/mol) unit.
    
    Parameters
    ----------
    dcct_hba1c : float
        HbA1c value in DCCT percentage (%).
        The DCCT (Diabetes Control and Complications Trial) percentage is commonly used in some regions to
        represent glycemic control.
        
    Returns
    -------
    ifcc_hba1c : float
        HbA1c value in IFCC unit (mmol/mol).
        The IFCC (International Federation of Clinical Chemistry and Laboratory Medicine) unit is an internationally
        standardized measure of HbA1c.
        
    Notes
    -----
    The conversion formula is as follows:
    
    IFCC-HbA1c (mmol/mol) = [DCCT-HbA1c (%) - 2.15] * 10.929
    
    This function uses the above formula to convert HbA1c values from the DCCT standard to the IFCC standard.
    
    Reference
    ---------
    The formula is derived from the relationship between the DCCT and IFCC HbA1c standards as defined by the
    International Federation of Clinical Chemistry.
    """
    return (dcct_hba1c - 2.15) * 10.929