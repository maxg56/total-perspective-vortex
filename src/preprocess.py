"""
Preprocessing module for EEG data from Physionet EEGMMIDB dataset.

Handles:
- Data loading from Physionet
- Bandpass filtering (7-30 Hz for motor imagery)
- Epoch extraction around motor events
"""

import os
import numpy as np
import mne
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf


# Physionet EEGMMIDB run descriptions:
# Runs 1, 2: Baseline (eyes open, eyes closed)
# Runs 3, 7, 11: Motor execution (left fist, right fist)
# Runs 4, 8, 12: Motor imagery (left fist, right fist)
# Runs 5, 9, 13: Motor execution (both fists, both feet)
# Runs 6, 10, 14: Motor imagery (both fists, both feet)

# Event mapping for motor imagery tasks
EVENT_ID_HANDS_FEET = {
    'T1': 1,  # hands (both fists)
    'T2': 2,  # feet (both feet)
}

EVENT_ID_LEFT_RIGHT = {
    'T1': 1,  # left fist
    'T2': 2,  # right fist
}


def get_data_path():
    """Get the path where MNE stores downloaded data."""
    return os.path.join(mne.get_config('MNE_DATA', default='~/mne_data'), 'MNE-eegbci-data')


def load_raw_data(subject: int, runs: list) -> mne.io.Raw:
    """
    Load raw EEG data for a specific subject and runs from Physionet EEGMMIDB.

    Parameters
    ----------
    subject : int
        Subject number (1-109)
    runs : list
        List of run numbers to load

    Returns
    -------
    raw : mne.io.Raw
        Concatenated raw EEG data
    """
    # Download and load the data
    raw_fnames = eegbci.load_data(subject, runs)
    raws = [read_raw_edf(f, preload=True, verbose=False) for f in raw_fnames]

    # Concatenate if multiple runs
    if len(raws) > 1:
        raw = concatenate_raws(raws)
    else:
        raw = raws[0]

    # Standardize channel names to 10-20 system
    eegbci.standardize(raw)

    # Set montage for channel locations
    montage = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(montage)

    return raw


def filter_raw(raw: mne.io.Raw, l_freq: float = 7.0, h_freq: float = 30.0) -> mne.io.Raw:
    """
    Apply bandpass filter to raw EEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    l_freq : float
        Low cutoff frequency (default: 7 Hz for mu rhythm)
    h_freq : float
        High cutoff frequency (default: 30 Hz for beta rhythm)

    Returns
    -------
    raw : mne.io.Raw
        Filtered raw data
    """
    raw.filter(l_freq, h_freq, fir_design='firwin', skip_by_annotation='edge', verbose=False)
    return raw


def extract_epochs(raw: mne.io.Raw, event_id: dict = None,
                   tmin: float = 0.0, tmax: float = 3.0,
                   baseline: tuple = None) -> mne.Epochs:
    """
    Extract epochs from raw EEG data around events.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    event_id : dict
        Dictionary mapping event names to event codes
    tmin : float
        Start time of epoch relative to event (default: 0)
    tmax : float
        End time of epoch relative to event (default: 3s)
    baseline : tuple
        Baseline correction period (default: None)

    Returns
    -------
    epochs : mne.Epochs
        Extracted epochs
    """
    # Get events from annotations
    events, event_dict = mne.events_from_annotations(raw, verbose=False)

    # Use provided event_id or extract from annotations
    if event_id is None:
        # Filter to keep only T1 and T2 (task events)
        event_id = {k: v for k, v in event_dict.items() if k in ['T1', 'T2']}

    # Pick only EEG channels
    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')

    # Create epochs
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                        picks=picks, baseline=baseline, preload=True, verbose=False)

    return epochs


def get_run_type(run: int) -> str:
    """
    Determine the type of motor task for a given run number.

    Parameters
    ----------
    run : int
        Run number

    Returns
    -------
    str
        'hands_feet' for both fists vs both feet tasks
        'left_right' for left fist vs right fist tasks
        'baseline' for baseline runs
    """
    if run in [1, 2]:
        return 'baseline'
    elif run in [3, 7, 11, 4, 8, 12]:
        return 'left_right'
    elif run in [5, 9, 13, 6, 10, 14]:
        return 'hands_feet'
    else:
        raise ValueError(f"Invalid run number: {run}")


def preprocess_subject(subject: int, runs: list,
                       l_freq: float = 7.0, h_freq: float = 30.0,
                       tmin: float = 0.0, tmax: float = 3.0) -> tuple:
    """
    Complete preprocessing pipeline for a subject.

    Parameters
    ----------
    subject : int
        Subject number (1-109)
    runs : list
        List of run numbers to load
    l_freq : float
        Low cutoff frequency for bandpass filter
    h_freq : float
        High cutoff frequency for bandpass filter
    tmin : float
        Start time of epoch relative to event
    tmax : float
        End time of epoch relative to event

    Returns
    -------
    X : np.ndarray
        EEG data array of shape (n_epochs, n_channels, n_times)
    y : np.ndarray
        Labels array of shape (n_epochs,)
    epochs : mne.Epochs
        The epochs object for further analysis
    """
    # Load raw data
    raw = load_raw_data(subject, runs)

    # Apply bandpass filter
    raw = filter_raw(raw, l_freq, h_freq)

    # Determine event mapping based on run type
    run_type = get_run_type(runs[0])
    if run_type == 'hands_feet':
        event_id = EVENT_ID_HANDS_FEET
    elif run_type == 'left_right':
        event_id = EVENT_ID_LEFT_RIGHT
    else:
        raise ValueError("Cannot preprocess baseline runs for classification")

    # Extract epochs
    epochs = extract_epochs(raw, event_id=event_id, tmin=tmin, tmax=tmax)

    # Get data and labels
    X = epochs.get_data()  # (n_epochs, n_channels, n_times)
    y = epochs.events[:, -1]  # Labels from event codes

    return X, y, epochs


def load_multiple_subjects(subjects: list, runs: list,
                           l_freq: float = 7.0, h_freq: float = 30.0,
                           tmin: float = 0.0, tmax: float = 3.0) -> tuple:
    """
    Load and preprocess data from multiple subjects.

    Parameters
    ----------
    subjects : list
        List of subject numbers
    runs : list
        List of run numbers to load for each subject
    l_freq : float
        Low cutoff frequency for bandpass filter
    h_freq : float
        High cutoff frequency for bandpass filter
    tmin : float
        Start time of epoch relative to event
    tmax : float
        End time of epoch relative to event

    Returns
    -------
    X : np.ndarray
        Concatenated EEG data array
    y : np.ndarray
        Concatenated labels array
    """
    X_all = []
    y_all = []

    for subject in subjects:
        try:
            X, y, _ = preprocess_subject(subject, runs, l_freq, h_freq, tmin, tmax)
            X_all.append(X)
            y_all.append(y)
            print(f"Subject {subject}: {len(y)} epochs loaded")
        except Exception as e:
            print(f"Error loading subject {subject}: {e}")
            continue

    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)

    return X, y


if __name__ == "__main__":
    # Example usage
    print("Testing preprocessing pipeline...")

    # Load subject 1, run 6 (motor imagery: hands vs feet)
    X, y, epochs = preprocess_subject(subject=1, runs=[6])

    print(f"Data shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Unique labels: {np.unique(y)}")
    print(f"Class distribution: {np.bincount(y)[1:]}")
