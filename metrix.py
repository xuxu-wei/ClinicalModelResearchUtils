import numpy as np
import pandas as pd
from math import ceil
from tqdm import tqdm
from sklearn.utils import resample

def get_time_range(times: pd.Series, lower=1, upper=99.95, step=1):
    """
    Determine a range of time values based on specified percentiles of survival times.

    Parameters
    ----------
    times : pd.Series
        Series containing survival times, dtype=float.
    lower : float
        Lower percentile for the time range.
    upper : float
        Upper percentile for the time range.
    step : int or float, default 1
        Step size for the time range.

    Returns
    -------
    predict_time_range : np.ndarray
        Array representing the range of time values based on specified percentiles.
    """
    lower_bound, upper_bound = np.percentile(times, [lower, upper])
    predict_time_range = np.arange(ceil(lower_bound), int(upper_bound + 1), step)
    return predict_time_range

def check_data_validity(y_true_surv, y_true_event, y_pred):
    """
    Validates data to ensure that it is appropriate for C-index calculation.

    Parameters
    ----------
    y_true_surv : array-like
        Array of actual survival times.
    y_true_event : array-like
        Array of event indicators (1 if the event occurred, 0 otherwise).
    y_pred : array-like
        Array of predicted risks or hazards.

    Returns
    -------
    bool
        Returns True if data is valid for C-index calculation, False otherwise.
    """
    if len(y_true_surv) < 2:
        return False  # Insufficient data

    if np.all(y_true_event == 0) or np.all(y_true_event == 1):
        return False  # Only one event status present in data

    if len(np.unique(y_true_surv)) == 1:
        return False  # Lack of diversity in survival times

    if len(np.unique(y_pred)) == 1:
        return False  # Lack of diversity in predicted values

    return True

def get_continuous_nri(y_true, old_model_preds, new_model_preds):
    """
    Calculate the Continuous Net Reclassification Improvement (NRI).

    Parameters
    ----------
    y_true : array-like
        True event labels (1 if the event occurred, 0 otherwise).
    old_model_preds : array-like
        Predictions from the old model (probabilities or risk scores).
    new_model_preds : array-like
        Predictions from the new model (probabilities or risk scores).

    Returns
    -------
    tuple
        (nri_event, nri_nonevent, continuous_nri) containing the continuous NRI values.
    """
    y_true = np.array(y_true)
    old_model_preds = np.array(old_model_preds)
    new_model_preds = np.array(new_model_preds)
    
    # Indices where events occurred (positive cases) and did not occur (negative cases)
    event_indices = y_true == 1
    nonevent_indices = y_true == 0
    
    # Proportion where the new model scores higher for events and lower for non-events
    P_up_event = np.mean(new_model_preds[event_indices] > old_model_preds[event_indices])
    P_down_event = np.mean(new_model_preds[event_indices] < old_model_preds[event_indices])
    P_down_nonevent = np.mean(new_model_preds[nonevent_indices] < old_model_preds[nonevent_indices])
    P_up_nonevent = np.mean(new_model_preds[nonevent_indices] > old_model_preds[nonevent_indices])
    
    # Continuous NRI calculation
    nri_event = P_up_event - P_down_event
    nri_nonevent = P_down_nonevent - P_up_nonevent
    continuous_nri = nri_event + nri_nonevent
    
    return nri_event, nri_nonevent, continuous_nri

def classify_by_thresholds(probs, thresholds):
    """
    Classify probabilities into multiple categories based on specified thresholds.

    Parameters
    ----------
    probs : array-like
        Model-predicted probabilities.
    thresholds : array-like
        List of threshold values for classification (e.g., [0.2, 0.5]).

    Returns
    -------
    array-like
        List of category indices after classification.
    """
    return np.digitize(probs, thresholds)

def get_categorical_nri_detail(y_true, old_model_preds, new_model_preds, thresholds):
    """
    Calculate Categorical NRI and return indices of reclassified samples.

    Parameters
    ----------
    y_true : array-like or pd.Series
        True event labels (1 if the event occurred, 0 otherwise).
    old_model_preds : array-like
        Predictions from the old model (probabilities or risk scores).
    new_model_preds : array-like
        Predictions from the new model (probabilities or risk scores).
    thresholds : array-like
        List of threshold values for risk categories.

    Returns
    -------
    dict
        Dictionary containing reclassification indices and NRI values.
    """
    old_model_preds = np.array(old_model_preds)
    new_model_preds = np.array(new_model_preds)
    
    # Keep the original indices if y_true is a pd.Series
    if isinstance(y_true, pd.Series):
        indices = y_true.index
        y_true = y_true.values
    else:
        indices = np.arange(len(y_true))
    
    # Classify scores by threshold
    old_class = classify_by_thresholds(old_model_preds, thresholds)
    new_class = classify_by_thresholds(new_model_preds, thresholds)
    
    # Event and non-event indices
    event_indices = y_true == 1
    nonevent_indices = y_true == 0
    
    # Dictionary for storing reclassified sample indices
    reclassification_indices = {
        'up_event': indices[event_indices & (new_class > old_class)],
        'down_event': indices[event_indices & (new_class < old_class)],
        'up_nonevent': indices[nonevent_indices & (new_class > old_class)],
        'down_nonevent': indices[nonevent_indices & (new_class < old_class)]
    }
    
    # Calculate the proportions for reclassification
    up_event = np.mean(new_class[event_indices] > old_class[event_indices])
    down_event = np.mean(new_class[event_indices] < old_class[event_indices])
    up_nonevent = np.mean(new_class[nonevent_indices] > old_class[nonevent_indices])
    down_nonevent = np.mean(new_class[nonevent_indices] < old_class[nonevent_indices])
    
    # Count reclassified samples
    reclassification_indices['num_up_event'] = np.sum(new_class[event_indices] > old_class[event_indices])
    reclassification_indices['num_down_event'] = np.sum(new_class[event_indices] < old_class[event_indices])
    reclassification_indices['num_up_nonevent'] = np.sum(new_class[nonevent_indices] > old_class[nonevent_indices])
    reclassification_indices['num_down_nonevent'] = np.sum(new_class[nonevent_indices] < old_class[nonevent_indices])
    
    # Categorical NRI calculation
    nri_event = up_event - down_event
    nri_nonevent = down_nonevent - up_nonevent
    categorical_nri = nri_event + nri_nonevent

    # For binary classification, construct a contingency table
    if len(np.unique(y_true)) == 2:
        total_event_count = np.sum(event_indices)
        total_nonevent_count = np.sum(nonevent_indices)
        event_num_dict = {'event': total_event_count, 'nonevent': total_nonevent_count}
        
        # Helper for formatting table row names
        index_formatter = lambda x: f'{x} (n = {event_num_dict[x]:,})'
        
        # MultiIndex for cross table rows and columns
        index = pd.MultiIndex.from_tuples([(index_formatter(event_status), 'old class', c)
                                           for event_status in ['event', 'nonevent'] for c in [*np.unique(y_true), 'total']], names=['', '', ''])
        columns = pd.MultiIndex.from_tuples([('new class', c) for c in np.unique(y_true)] + [('', 'total')], names=['', ''])

        df_nri_cross_table = pd.DataFrame(index=index, columns=columns)
        sorted_class = sorted(np.unique(y_true), reverse=False)

        # Fill table with values
        for event_status in ['event', 'nonevent']:
            for c, change in zip(sorted_class, ['up', 'down']):
                df_nri_cross_table.loc[(index_formatter(event_status), 'old class', c), ('new class', c)] = len(reclassification_indices[f'stay_{c}_{event_status}'])
                neg_c = [i for i in sorted_class if i != c][0]
                df_nri_cross_table.loc[(index_formatter(event_status), 'old class', c), ('new class', neg_c)] = len(reclassification_indices[f'{change}_{event_status}'])

        for event_status in ['event', 'nonevent']:
            df_nri_cross_table.loc[(index_formatter(event_status), 'old class', 'total')] = df_nri_cross_table.loc[index_formatter(event_status)].sum()
            df_nri_cross_table.loc[index_formatter(event_status), ('', 'total')] = df_nri_cross_table.loc[index_formatter(event_status)].sum(axis=1).to_list()

        # Format table with percentages
        df_nri_cross_table_event = df_nri_cross_table.loc[[index_formatter('event')]].astype(int).applymap(lambda x: f'{x} ({x / total_event_count * 100:.2f}%)')
        df_nri_cross_table_nonevent = df_nri_cross_table.loc[[index_formatter('nonevent')]].astype(int).applymap(lambda x: f'{x} ({x / total_nonevent_count * 100:.2f}%)')
        df_nri_cross_table_format = pd.concat([df_nri_cross_table_event, df_nri_cross_table_nonevent])

        reclassification_indices['reclassification_table_format'] = df_nri_cross_table_format
        reclassification_indices['reclassification_table'] = df_nri_cross_table
        
    return nri_event, nri_nonevent, categorical_nri, reclassification_indices

def get_idi(y_true, old_model_preds, new_model_preds):
    """
    Calculate Integrated Discrimination Improvement (IDI).

    Parameters
    ----------
    y_true : array-like
        True event labels (1 if the event occurred, 0 otherwise).
    old_model_preds : array-like
        Predictions from the old model (probabilities or risk scores).
    new_model_preds : array-like
        Predictions from the new model (probabilities or risk scores).

    Returns
    -------
    float
        The IDI value.
    """
    y_true = np.array(y_true)
    old_model_preds = np.array(old_model_preds)
    new_model_preds = np.array(new_model_preds)
    
    # Indices for events and non-events
    event_indices = y_true == 1
    nonevent_indices = y_true == 0
    
    # Calculate mean differences between new and old model predictions for events and non-events
    delta_event = np.mean(new_model_preds[event_indices] - old_model_preds[event_indices])
    delta_nonevent = np.mean(new_model_preds[nonevent_indices] - old_model_preds[nonevent_indices])
    
    # IDI calculation
    idi = delta_event - delta_nonevent
    
    return idi

def bootstrap_resampler(*data, n_iterations=1000, show_progress=True, random_seed=19960816):
    """
    Generator that yields resampled datasets for each iteration.

    Parameters
    ----------
    data : tuple
        Data to resample (e.g., arrays or DataFrames).
    n_iterations : int, default 1000
        Number of bootstrap iterations.
    show_progress : bool, default True
        Display progress bar if True.
    random_seed : int, default 19960816
        Seed for reproducibility of resampling.

    Yields
    ------
    tuple
        Resampled data for each iteration.
    """
    iter_wrapper = tqdm(range(n_iterations)) if show_progress else range(n_iterations)
    for i in iter_wrapper:
        data_resampled = resample(*data, random_state=random_seed + i)
        yield data_resampled
