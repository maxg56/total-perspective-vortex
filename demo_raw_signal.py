#!/usr/bin/env python
"""
Demonstration script for raw and filtered EEG signal visualization.

This script demonstrates:
1. Loading raw EEG data from the Physionet EEGMMIDB dataset
2. Visualizing raw signal traces
3. Applying bandpass filtering (7-30 Hz for motor imagery)
4. Visualizing filtered signal traces
5. Computing and displaying Power Spectral Density (PSD) in the 7-30 Hz range

Usage:
    python demo_raw_signal.py [subject] [run]

Example:
    python demo_raw_signal.py 4 14
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import matplotlib.pyplot as plt
from constants import (
    DEFAULT_SUBJECT, DEFAULT_RUN, SEPARATOR_WIDTH_WIDE,
    EEG_BANDPASS_LOW, EEG_BANDPASS_HIGH
)
from preprocess import load_raw_data, filter_raw, get_run_type


def main():
    """Main function to demonstrate raw and filtered EEG signal visualization."""
    # Get subject and run from command line or use defaults
    subject = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SUBJECT
    run = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_RUN

    print("=" * SEPARATOR_WIDTH_WIDE)
    print("EEG Raw Signal Visualization Demo")
    print("=" * SEPARATOR_WIDTH_WIDE)
    print(f"Subject: {subject}")
    print(f"Run: {run}")
    print(f"Run type: {get_run_type(run)}")
    print()

    # Load raw data
    print("Loading raw EEG data from Physionet EEGMMIDB dataset...")
    raw = load_raw_data(subject, [run])
    print(f"Sampling rate: {raw.info['sfreq']} Hz")
    print(f"Number of channels: {len(raw.ch_names)}")
    print(f"Duration: {raw.times[-1]:.2f} seconds")
    print(f"Channels: {', '.join(raw.ch_names[:10])}...")
    print()

    # Create a copy for filtering
    raw_filtered = raw.copy()

    # Apply bandpass filter
    print(f"Applying bandpass filter: {EEG_BANDPASS_LOW}-{EEG_BANDPASS_HIGH} Hz")
    print("(Targeting mu [8-12 Hz] and beta [12-30 Hz] rhythms for motor imagery)")
    raw_filtered = filter_raw(raw_filtered, EEG_BANDPASS_LOW, EEG_BANDPASS_HIGH)
    print()

    # Visualization
    print("=" * SEPARATOR_WIDTH_WIDE)
    print("VISUALIZATIONS")
    print("=" * SEPARATOR_WIDTH_WIDE)
    print()

    # 1. Plot raw signal
    print("1. Displaying RAW signal (unfiltered)...")
    print("   Close the plot window to continue...")
    fig1 = raw.plot(
        duration=10.0,
        n_channels=30,
        scalings='auto',
        title=f'Raw EEG Signal - Subject {subject}, Run {run}',
        show=True,
        block=True
    )
    print()

    # 2. Plot filtered signal
    print("2. Displaying FILTERED signal (7-30 Hz bandpass)...")
    print("   Close the plot window to continue...")
    fig2 = raw_filtered.plot(
        duration=10.0,
        n_channels=30,
        scalings='auto',
        title=f'Filtered EEG Signal (7-30 Hz) - Subject {subject}, Run {run}',
        show=True,
        block=True
    )
    print()

    # 3. Plot Power Spectral Density (PSD)
    print("3. Computing and displaying Power Spectral Density (PSD)...")
    print("   Comparing raw vs filtered signals in the 7-30 Hz range...")
    print("   Close the plot window to continue...")

    # Create figure with two subplots for comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # PSD for raw signal
    raw.plot_psd(
        fmin=EEG_BANDPASS_LOW,
        fmax=EEG_BANDPASS_HIGH,
        average=True,
        spatial_colors=False,
        ax=axes[0],
        show=False
    )
    axes[0].set_title(f'PSD - Raw Signal\nSubject {subject}, Run {run}')
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Power Spectral Density (dB)')
    axes[0].grid(True, alpha=0.3)

    # PSD for filtered signal
    raw_filtered.plot_psd(
        fmin=EEG_BANDPASS_LOW,
        fmax=EEG_BANDPASS_HIGH,
        average=True,
        spatial_colors=False,
        ax=axes[1],
        show=False
    )
    axes[1].set_title(f'PSD - Filtered Signal (7-30 Hz)\nSubject {subject}, Run {run}')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Power Spectral Density (dB)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show(block=True)
    print()

    # 4. Plot topographic PSD maps
    print("4. Displaying topographic PSD maps...")
    print("   Showing spatial distribution of power in mu (8-12 Hz) and beta (12-30 Hz) bands...")
    print("   Close the plot window to continue...")

    fig_topo = raw_filtered.plot_psd_topo(
        tmax=60.0,  # Use first 60 seconds
        show=True,
        block=True
    )
    print()

    # Summary
    print("=" * SEPARATOR_WIDTH_WIDE)
    print("SUMMARY")
    print("=" * SEPARATOR_WIDTH_WIDE)
    print("This demonstration showed:")
    print("  1. Raw EEG signal traces (unfiltered)")
    print("  2. Filtered EEG signal traces (7-30 Hz bandpass)")
    print("  3. Power Spectral Density comparison (raw vs filtered)")
    print("  4. Topographic PSD maps showing spatial distribution")
    print()
    print("Key observations:")
    print("  - Bandpass filtering (7-30 Hz) targets mu and beta rhythms")
    print("  - Mu rhythm (8-12 Hz): Associated with motor cortex activity")
    print("  - Beta rhythm (12-30 Hz): Related to motor control and planning")
    print("  - These frequency bands are critical for motor imagery BCI")
    print("=" * SEPARATOR_WIDTH_WIDE)


if __name__ == "__main__":
    main()
