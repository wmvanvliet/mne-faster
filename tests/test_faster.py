"""Unit tests for the FASTER pipeline."""

import mne
import numpy as np
import pytest
from mne_faster import (
    find_bad_channels,
    find_bad_channels_in_epochs,
    find_bad_epochs,
    hurst,
)
from mne_faster.faster import _freqs_power
from numpy.testing import assert_allclose, assert_almost_equal

# Signal properties used in the tests
length = 2  # in seconds
srate = 200.0  # in Hertz
n_channels = 32
n_epochs = 100
n_samples = int(length * srate)
time = np.arange(n_samples) / srate

# Fix the seed
rng = np.random.RandomState(123)


def test_hurst():
    """Test internal hurst exponent function."""
    # Hurst exponent of a sine wave
    p = np.atleast_2d(np.sin(1000))
    assert_almost_equal(p, 0.82687954)

    # Positive first derivative, hurst > 1
    p = np.atleast_2d(np.log10(np.cumsum(rng.randn(1000) + 100)))
    assert hurst(p) > 1

    # First derivative alternating around zero, hurst ~ 0
    p = np.atleast_2d(np.log10(rng.randn(1000) + 1000))
    assert_allclose(hurst(p), 0, atol=0.1)

    # Positive, but fluctuating first derivative, hurst ~ 0.5
    p = np.atleast_2d(np.log10(np.cumsum(rng.randn(1000)) + 1000))
    assert_allclose(hurst(p), 0.5, atol=0.1)


# This function also implicitly tests _efficient_welch
def test_freqs_power():
    """Test internal function for frequency power estimation."""
    # Create signal with different frequency components
    freqs = [1, 5, 12.8, 23.4, 40]  # in Hertz
    srate = 100.0
    time = np.arange(10 * srate) / srate
    signal = np.sum([np.sin(2 * np.pi * f * time) for f in freqs], axis=0)
    signal = np.atleast_2d(signal)

    # These frequencies should be present
    for f in freqs:
        assert_almost_equal(_freqs_power(signal, srate, [f]), 3 + 1 / 3.0)

    # The function should sum the individual frequency  powers
    assert_almost_equal(_freqs_power(signal, srate, freqs), len(freqs) * (3 + 1 / 3.0))

    # These frequencies should not be present
    assert_almost_equal(_freqs_power(signal, srate, [2, 4, 13, 23, 35]), 0)

    # Insufficient sample rate to calculate this frequency
    with pytest.raises(ValueError):
        _freqs_power(signal, srate, [51])


def _baseline_signal():
    """Create the baseline signal."""
    signal = np.tile(np.sin(time), (n_epochs, n_channels, 1))
    noise = rng.randn(n_epochs, n_channels, n_samples)
    return signal, noise


def _to_epochs(signal, noise):
    """Create a testing epochs object."""
    events = np.tile(np.arange(n_epochs)[:, np.newaxis], (1, 3))
    return mne.EpochsArray(
        signal + noise, mne.create_info(n_channels, srate, "eeg"), events
    )


def test_find_bad_channels():
    """Test detecting bad channels through outlier detection."""
    signal, noise = _baseline_signal()

    # This channel has more noise
    noise[:, 0, :] *= 2

    # This channel does not correlate with the others
    signal[:, 1, :] = np.sin(time + 0.68)

    # This channel has excessive 50 Hz line noise
    signal[:, 2, :] += np.sin(50 * 2 * np.pi * time)

    # This channel has excessive 60 Hz line noise
    signal[:, 3, :] += 1.2 * np.sin(60 * 2 * np.pi * time)

    # This channel has a different noise signature (kurtosis)
    noise[:, 4, :] = 4 * rng.rand(n_epochs, n_samples)

    # TODO: deviant hurst
    epochs = _to_epochs(signal, noise)
    bads = find_bad_channels(epochs, max_iter=1, return_by_metric=True)
    assert bads == dict(
        variance=["0"],
        correlation=["1"],
        line_noise=["2", "3"],
        kurtosis=["4"],
        hurst=["2"],
    )

    # Test picks
    bads = find_bad_channels(epochs, return_by_metric=True, picks=range(3, n_channels))
    assert bads["line_noise"] == ["3"]


def test_find_bad_epochs():
    """Test detecting bad epochs through outlier detection."""
    signal, noise = _baseline_signal()

    # This epoch has more noise
    noise[0, :, :] *= 2

    # This epoch has some deviation
    signal[1, :, :] += 20

    # This epoch has a single spike across channels
    signal[2, :, 0] += 10

    epochs = _to_epochs(signal, noise)

    bads = find_bad_epochs(epochs, max_iter=1, return_by_metric=True)
    assert bads == dict(
        variance=[0],
        deviation=[1],
        amplitude=[0, 2],
    )

    # Test picks
    bads = find_bad_epochs(epochs, return_by_metric=True, picks=range(3, n_channels))
    assert bads == dict(
        variance=[0],
        deviation=[1],
        amplitude=[0, 2],
    )


def test_find_bad_channels_in_epochs():
    """Test detecting bad channels in each epoch through outlier detection."""
    signal, noise = _baseline_signal()

    # This channel/epoch combination has more noise
    noise[0, 0, :] *= 2

    # This channel/epoch combination has some deviation
    signal[1, 1, :] += 20

    # This channel/epoch combination has a single spike
    signal[2, 2, 0] += 100

    # This channel/epoch combination has excessive 50 Hz line noise
    signal[3, 3, :] += np.sin(50 * 2 * np.pi * time)

    epochs = _to_epochs(signal, noise)

    bads = find_bad_channels_in_epochs(epochs, return_by_metric=True, thres=5)
    assert bads["variance"][0] == ["0"]
    assert bads["deviation"][1] == ["1"]
    assert bads["amplitude"][2] == ["2"]
    assert bads["median_gradient"][0] == ["0"]
    assert bads["line_noise"][3] == ["3"]

    # Test picks
    bads = find_bad_channels_in_epochs(
        epochs, return_by_metric=True, thres=5, picks=range(3, n_channels)
    )
    assert bads["variance"][0] == []
    assert bads["deviation"][1] == []
    assert bads["amplitude"][2] == []
    assert bads["median_gradient"][0] == []
    assert bads["line_noise"][3] == ["3"]


def test_distance_correction():
    """Test correcting for the distance of each electrode to the reference."""
    signal, noise = _baseline_signal()
    epochs = _to_epochs(signal, noise)
    for ch in range(32):
        # Set electrode position and reference position
        epochs.info["chs"][ch]["loc"] = np.array(
            [
                np.sin(ch * np.pi / 32.0),  # x
                np.cos(ch * np.pi / 32.0),  # y
                0.0,  # z
                0.0,
                1.0,
                0.0,  # x,y,z for reference electrode
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )

        # Make the signal amplitude increase with the distance to the sensor.
        # This is not noise, but a natural phenomenon.
        epochs._data[:, ch, :] *= ch

    # Add extra noise to channel 3 for all epochs
    epochs._data[:, 3, :] *= 5

    # Add extra noise to channel 5 for epoch 3
    epochs._data[3, 5, :] *= 5

    # Without distance correction, channel 3 is not marked as bad
    bads = find_bad_channels(
        epochs,
        eeg_ref_corr=False,
        use_metrics=["variance"],
    )
    assert bads == []

    # With distance correction, channel 3 is correctly found
    bads = find_bad_channels(
        epochs,
        eeg_ref_corr=True,
        use_metrics=["variance"],
    )
    assert bads == ["3"]

    # Without distance correction, channel 5 is not marked as bad in epoch 3
    bads = find_bad_channels_in_epochs(
        epochs,
        eeg_ref_corr=False,
        use_metrics=["variance"],
    )
    assert bads[3] == []

    # With distance correction, channel 5 is correctly found in epoch 3
    bads = find_bad_channels_in_epochs(
        epochs,
        eeg_ref_corr=True,
        use_metrics=["variance"],
    )
    assert bads[3] == ["5"]
