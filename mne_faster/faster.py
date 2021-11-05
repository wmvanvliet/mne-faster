from collections import defaultdict
import numpy as np
import scipy.signal
import mne
from scipy.stats import kurtosis
from mne.preprocessing.bads import _find_outliers
from mne.utils import logger
from mne.io.pick import pick_info, _picks_by_type


def _bad_mask_to_names(info, bad_mask):
    """Remap mask to ch names"""
    bad_idx = [np.where(m)[0] for m in bad_mask]
    return [[info['ch_names'][k] for k in epoch] for epoch in bad_idx]


def _combine_indices(bads):
    """summarize indices"""
    return list(set(v for val in bads.values() if len(val) > 0 for v in val))


def hurst(x):
    """Estimate Hurst exponent on a timeseries.

    The estimation is based on the second order discrete derivative.

    Parameters
    ----------
    x : 1D numpy array
        The timeseries to estimate the Hurst exponent for.

    Returns
    -------
    h : float
        The estimation of the Hurst exponent for the given timeseries.
    """
    y = np.cumsum(np.diff(x, axis=1), axis=1)

    b1 = [1, -2, 1]
    b2 = [1, 0, -2, 0, 1]

    # second order derivative
    y1 = scipy.signal.lfilter(b1, 1, y, axis=1)
    y1 = y1[:, len(b1) - 1:-1]  # first values contain filter artifacts

    # wider second order derivative
    y2 = scipy.signal.lfilter(b2, 1, y, axis=1)
    y2 = y2[:, len(b2) - 1:-1]  # first values contain filter artifacts

    s1 = np.mean(y1 ** 2, axis=1)
    s2 = np.mean(y2 ** 2, axis=1)

    return 0.5 * np.log2(s2 / s1)


def _efficient_welch(data, sfreq):
    """Calls scipy.signal.welch with parameters optimized for greatest speed
    at the expense of precision. The window is set to ~10 seconds and windows
    are non-overlapping.
    Parameters
    ----------
    data : array, shape (..., n_samples)
        The timeseries to estimate signal power for. The last dimension
        is assumed to be time.
    sfreq : float
        The sample rate of the timeseries.
    Returns
    -------
    fs : array of float
        The frequencies for which the power spectra was calculated.
    ps : array, shape (..., frequencies)
        The power spectra for each timeseries.
    """
    from scipy.signal import welch
    nperseg = min(data.shape[-1],
                  2 ** int(np.log2(10 * sfreq) + 1))  # next power of 2

    return welch(data, sfreq, nperseg=nperseg, noverlap=0, axis=-1)


def _freqs_power(data, sfreq, freqs):
    fs, ps = _efficient_welch(data, sfreq)
    try:
        return np.sum([ps[..., np.searchsorted(fs, f)] for f in freqs], axis=0)
    except IndexError:
        raise ValueError(
            ("Insufficient sample rate to  estimate power at {} Hz for line "
             "noise detection. Use the 'metrics' parameter to disable the "
             "'line_noise' metric.").format(freqs))


def _distance_correction(info, picks, x):
    """Remove the effect of distance to reference sensor.

    Computes the distance of each sensor to the reference sensor. Then
    regresses the effect of this distance out of the values in x.

    Parameters
    ----------
    info : instance of Info
        The measurement info. This should contain positions for all the
        sensors.
    picks : list of int
        Indices of the channels that correspond to the values in x.
    x : list of float
        Values to correct.

    Returns
    -------
    x_corr : list of float
        values in x corrected for the distance to reference sensor.
    """
    pos = np.array([info['chs'][ch]['loc'][:3] for ch in picks])
    ref_pos = np.array([info['chs'][ch]['loc'][3:6] for ch in picks])

    if np.any(np.all(pos == 0, axis=1)):
        raise ValueError('Cannot perform correction for distance to reference '
                         'sensor: not all selected channels have position '
                         'information.')
    if np.any(np.all(ref_pos == 0, axis=1)):
        raise ValueError('Cannot perform correction for distance to reference '
                         'sensor: the location of the reference sensor is not '
                         'specified for all selected channels.')

    # Compute angular distances to the reference sensor
    pos /= np.linalg.norm(pos, axis=1)[:, np.newaxis]
    ref_pos /= np.linalg.norm(ref_pos, axis=1)[:, np.newaxis]
    angles = [np.arccos(np.dot(a, b)) for a, b in zip(pos, ref_pos)]

    # Fit a quadratic curve to correct for the angular distance
    fit = np.polyfit(angles, x, 2)
    return x - np.polyval(fit, angles)


def find_bad_channels(epochs, picks=None, max_iter=1, thres=3,
                      eeg_ref_corr=False, use_metrics=None,
                      return_by_metric=False):
    """Automatically find and mark bad channels.

    Implements the first step of the FASTER algorithm.

    This function attempts to automatically mark bad EEG channels by performing
    outlier detection. It operated on epoched data, to make sure only relevant
    data is analyzed.

    Parameters
    ----------
    epochs : Instance of Epochs
        The epochs for which bad channels need to be marked
    picks : list of int | None
        Channels to operate on. Defaults to EEG channels.
    thres : float
        The threshold value, in standard deviations, to apply. A channel
        crossing this threshold value is marked as bad. Defaults to 3.
    max_iter : int
        The maximum number of iterations performed during outlier detection
        (defaults to 1, as in the original FASTER paper).
    eeg_ref_corr : bool
        If the EEG data has been referenced using a single electrode setting
        this parameter to True will enable a correction factor for the distance
        of each electrode to the reference. If an average reference is applied,
        or the mean of multiple reference electrodes, set this parameter to
        False. Defaults to False, which disables the correction.
    use_metrics : list of str
        List of metrics to use. Can be any combination of:
            'variance', 'correlation', 'hurst', 'kurtosis', 'line_noise'
        Defaults to all of them.
    return_by_metric : bool
        Whether to return the bad channels as a flat list (False, default) or
        as a dictionary with the names of the used metrics as keys and the
        bad channels found by this metric as values.

    Returns
    -------
    bads : list of str
        The names of the bad EEG channels.
    """
    metrics = {
        'variance': lambda x: np.var(x, axis=1),
        'correlation': lambda x: np.nanmean(
            np.ma.masked_array(
                np.corrcoef(x),
                np.identity(len(x), dtype=bool)
            ),
            axis=0),
        'hurst': lambda x: hurst(x),
        'kurtosis': lambda x: kurtosis(x, axis=1),
        'line_noise': lambda x: _freqs_power(x, epochs.info['sfreq'],
                                             [50, 60]),
    }

    if picks is None:
        picks = mne.pick_types(epochs.info, meg=False, eeg=True, exclude=[])
    if use_metrics is None:
        use_metrics = metrics.keys()

    # Concatenate epochs in time
    data = epochs.get_data()[:, picks]
    data = data.transpose(1, 0, 2).reshape(data.shape[1], -1)

    # Find bad channels
    bads = defaultdict(list)
    info = pick_info(epochs.info, picks, copy=True)
    for ch_type, chs in _picks_by_type(info):
        logger.info('Bad channel detection on %s channels:' % ch_type.upper())
        for metric in use_metrics:
            scores = metrics[metric](data[chs])
            if eeg_ref_corr:
                scores = _distance_correction(epochs.info, picks, scores)
            bad_channels = [epochs.ch_names[picks[chs[i]]]
                            for i in _find_outliers(scores, thres, max_iter)]
            logger.info('\tBad by %s: %s' % (metric, bad_channels))
            bads[metric].append(bad_channels)

    bads = dict((k, np.concatenate(v).tolist()) for k, v in bads.items())

    if return_by_metric:
        return bads
    else:
        return _combine_indices(bads)


def _deviation(data):
    """Computes the deviation from mean for each channel in a set of epochs.

    This is not implemented as a lambda function, because the channel means
    should be cached during the computation.

    Parameters
    ----------
    data : 3D numpy array
        The epochs (#epochs x #channels x #samples).

    Returns
    -------
    dev : list of float
        For each epoch, the mean deviation of the channels.
    """
    ch_mean = np.mean(data, axis=2)
    return ch_mean - np.mean(ch_mean, axis=0)


def find_bad_epochs(epochs, picks=None, thres=3, max_iter=1, use_metrics=None,
                    return_by_metric=False):
    """Automatically find and mark bad epochs.

    Implements the second step of the FASTER algorithm.

    This function attempts to automatically mark bad epochs by performing
    outlier detection.

    Parameters
    ----------
    epochs : Instance of Epochs
        The epochs to analyze.
    picks : list of int | None
        Channels to operate on. Defaults to EEG channels.
    thres : float
        The threshold value, in standard deviations, to apply. An epoch
        crossing this threshold value is marked as bad. Defaults to 3.
    max_iter : int
        The maximum number of iterations performed during outlier detection
        (defaults to 1, as in the original FASTER paper).
    use_metrics : list of str
        List of metrics to use. Can be any combination of:
            'amplitude', 'variance', 'deviation'
        Defaults to all of them.
    return_by_metric : bool
        Whether to return the bad channels as a flat list (False, default) or
        as a dictionary with the names of the used metrics as keys and the
        bad channels found by this metric as values.

    Returns
    -------
    bads : list of int
        The indices of the bad epochs.
    """

    metrics = {
        'amplitude': lambda x: np.mean(np.ptp(x, axis=2), axis=1),
        'deviation': lambda x: np.mean(_deviation(x), axis=1),
        'variance': lambda x: np.mean(np.var(x, axis=2), axis=1),
    }

    if picks is None:
        picks = mne.pick_types(epochs.info, meg=False, eeg=True,
                               exclude='bads')
    if use_metrics is None:
        use_metrics = metrics.keys()

    info = pick_info(epochs.info, picks, copy=True)
    data = epochs.get_data()[:, picks]

    bads = defaultdict(list)
    for ch_type, chs in _picks_by_type(info):
        logger.info('Bad epoch detection on %s channels:' % ch_type.upper())
        for metric in use_metrics:
            scores = metrics[metric](data[:, chs])
            bad_epochs = _find_outliers(scores, thres, max_iter)
            logger.info('\tBad by %s: %s' % (metric, bad_epochs))
            bads[metric].append(bad_epochs)

    bads = dict((k, np.concatenate(v).tolist()) for k, v in bads.items())
    if return_by_metric:
        return bads
    else:
        return _combine_indices(bads)


def _power_gradient(data, sfreq, prange):
    """Estimate the gradient of the power spectrum at upper frequencies.

    Parameters
    ----------
    data : array, shape (n_components, n_samples)
        The timeseries to estimate signal power for. The last dimension
        is presumed to be time.
    sfreq : float
        The sample rate of the timeseries.
    prange : pair of floats
        The (lower, upper) frequency limits of the power spectrum to use. In
        the FASTER paper, they set these to the passband of the lowpass filter.

    Returns
    -------
    grad : array of float
        The gradients of the timeseries.
    """
    fs, ps = _efficient_welch(data, sfreq)

    # Limit power spectrum to selected frequencies
    start, stop = (np.searchsorted(fs, p) for p in prange)
    if start >= ps.shape[1]:
        raise ValueError(("Sample rate insufficient to estimate {} Hz power. "
                          "Use the 'power_gradient_range' parameter to tweak "
                          "the tested frequencies for this metric or use the "
                          "'metrics' parameter to disable the "
                          "'power_gradient' metric.").format(prange[0]))
    ps = ps[:, start:stop]

    # Compute mean gradients
    return np.mean(np.diff(ps), axis=1)


def find_bad_components(ica, epochs, thres=3, max_iter=1, use_metrics=None,
                        prange=None, return_by_metric=False):
    """Implements the third step of the FASTER algorithm.

    This function attempts to automatically mark bad ICA components by
    performing outlier detection.

    Parameters
    ----------
    ica : Instance of ICA
        The ICA operator, already fitted to the supplied Epochs object.
    epochs : Instance of Epochs
        The untransformed epochs to analyze.
    thres : float
        The threshold value, in standard deviations, to apply. A component
        crossing this threshold value is marked as bad. Defaults to 3.
    max_iter : int
        The maximum number of iterations performed during outlier detection
        (defaults to 1, as in the original FASTER paper).
    use_metrics : list of str
        List of metrics to use. Can be any combination of:
            'eog_correlation', 'kurtosis', 'power_gradient', 'hurst',
            'median_gradient'
        Defaults to all of them.
    prange : None | pair of floats
        The (lower, upper) frequency limits of the power spectrum to use for
        the power gradient computation. In the FASTER paper, they set these to
        the passband of the highpass and lowpass filter. If None, defaults to
        the 'highpass' and 'lowpass' filter settings in ica.info.
    return_by_metric : bool
        Whether to return the bad channels as a flat list (False, default) or
        as a dictionary with the names of the used metrics as keys and the
        bad channels found by this metric as values.

    Returns
    -------
    bads : list of int
        The indices of the bad components.

    See also
    --------
    ICA.find_bads_ecg
    ICA.find_bads_eog
    """
    source_data = ica.get_sources(epochs).get_data().transpose(1, 0, 2)
    source_data = source_data.reshape(source_data.shape[0], -1)

    if prange is None:
        prange = (ica.info['highpass'], ica.info['lowpass'])
    if len(prange) != 2:
        raise ValueError('prange must be a pair of floats')

    metrics = {
        'eog_correlation': lambda x: x.find_bads_eog(epochs)[1],
        'kurtosis': lambda x: kurtosis(
            np.dot(
                x.mixing_matrix_.T,
                x.pca_components_[:x.n_components_]),
            axis=1),
        'power_gradient': lambda x: _power_gradient(source_data,
                                                    ica.info['sfreq'],
                                                    prange),
        'hurst': lambda x: hurst(source_data),
        'median_gradient': lambda x: np.median(np.abs(np.diff(source_data)),
                                               axis=1),
        'line_noise': lambda x: _freqs_power(source_data,
                                             epochs.info['sfreq'], [50, 60]),
    }

    if use_metrics is None:
        use_metrics = metrics.keys()

    bads = defaultdict(list)
    for metric in use_metrics:
        scores = np.atleast_2d(metrics[metric](ica))
        for s in scores:
            bad_comps = _find_outliers(s, thres, max_iter)
            logger.info('Bad by %s:\n\t%s' % (metric, bad_comps))
            bads[metric].append(bad_comps)

    bads = dict((k, np.concatenate(v).tolist()) for k, v in bads.items())
    if return_by_metric:
        return bads
    else:
        return _combine_indices(bads)


def find_bad_channels_in_epochs(epochs, picks=None, thres=3, max_iter=1,
                                eeg_ref_corr=False, use_metrics=None,
                                return_by_metric=False):
    """Implements the fourth step of the FASTER algorithm.

    This function attempts to automatically mark bad channels in each epochs by
    performing outlier detection.

    Parameters
    ----------
    epochs : Instance of Epochs
        The epochs to analyze.
    picks : list of int | None
        Channels to operate on. Defaults to EEG channels.
    thres : float
        The threshold value, in standard deviations, to apply. An epoch
        crossing this threshold value is marked as bad. Defaults to 3.
    max_iter : int
        The maximum number of iterations performed during outlier detection
        (defaults to 1, as in the original FASTER paper).
    eeg_ref_corr : bool
        If the EEG data has been referenced using a single electrode setting
        this parameter to True will enable a correction factor for the distance
        of each electrode to the reference. If an average reference is applied,
        or the mean of multiple reference electrodes, set this parameter to
        False. Defaults to False, which disables the correction.
    use_metrics : list of str
        List of metrics to use. Can be any combination of:
            'amplitude', 'variance', 'deviation', 'median_gradient'
        Defaults to all of them.
    return_by_metric : bool
        Whether to return the bad channels as a flat list (False, default) or
        as a dictionary with the names of the used metrics as keys and the
        bad channels found by this metric as values.

    Returns
    -------
    bads : list of lists of int
        For each epoch, the indices of the bad channels.
    """

    metrics = {
        'amplitude': lambda x: np.ptp(x, axis=2),
        'deviation': lambda x: _deviation(x),
        'variance': lambda x: np.var(x, axis=2),
        'median_gradient': lambda x: np.median(np.abs(np.diff(x)), axis=2),
        'line_noise': lambda x: _freqs_power(x, epochs.info['sfreq'],
                                             [50, 60]),
    }

    if picks is None:
        picks = mne.pick_types(epochs.info, meg=False, eeg=True,
                               exclude='bads')
    if use_metrics is None:
        use_metrics = metrics.keys()

    info = pick_info(epochs.info, picks, copy=True)
    data = epochs.get_data()[:, picks]
    bads = dict((m, np.zeros((len(data), len(picks)), dtype=bool)) for
                m in metrics)
    for ch_type, chs in _picks_by_type(info):
        ch_names = [info['ch_names'][k] for k in chs]
        chs = np.array(chs)
        for metric in use_metrics:
            logger.info('Bad channel-in-epoch detection on %s channels:'
                        % ch_type.upper())
            s_epochs = metrics[metric](data[:, chs])
            for i_epochs, scores in enumerate(s_epochs):
                if eeg_ref_corr:
                    scores = _distance_correction(epochs.info, picks, scores)
                outliers = _find_outliers(scores, thres, max_iter)
                if len(outliers) > 0:
                    bad_segment = [ch_names[k] for k in outliers]
                    logger.info('Epoch %d, Bad by %s:\n\t%s' % (
                        i_epochs, metric, bad_segment))
                    bads[metric][i_epochs, chs[outliers]] = True

    info = pick_info(epochs.info, picks, copy=True)
    if return_by_metric:
        bads = dict((m, _bad_mask_to_names(info, v)) for m, v in bads.items())
    else:
        bads = np.sum(list(bads.values()), axis=0).astype(bool)
        bads = _bad_mask_to_names(info, bads)

    return bads


def run_faster(epochs, thres=3, copy=True):
    """Run the entire FASTER pipeline on the data.
    """
    if copy:
        epochs = epochs.copy()

    # Step one
    logger.info('Step 1: mark bad channels')
    epochs.info['bads'] += find_bad_channels(epochs, thres=5)

    # Step two
    logger.info('Step 2: mark bad epochs')
    bad_epochs = find_bad_epochs(epochs, thres=thres)
    good_epochs = list(set(range(len(epochs))).difference(set(bad_epochs)))
    epochs = epochs[good_epochs]

    # Step three (using the build-in MNE functionality for this)
    logger.info('Step 3: mark bad ICA components')
    picks = mne.pick_types(epochs.info, meg=False, eeg=True, eog=True,
                           exclude='bads')
    ica = mne.preprocessing.run_ica(epochs, len(picks), picks=picks,
                                    eog_ch=['vEOG', 'hEOG'])
    ica.apply(epochs)

    # Step four
    logger.info('Step 4: mark bad channels for each epoch')
    bad_channels_per_epoch = find_bad_channels_in_epochs(epochs, thres=thres)
    for i, b in enumerate(bad_channels_per_epoch):
        if len(b) > 0:
            epoch = epochs[i]
            epoch.info['bads'] += b
            epoch.interpolate_bads_eeg()
            epochs._data[i, :, :] = epoch._data[0, :, :]

    # Now that the data is clean, apply average reference
    epochs.info['custom_ref_applied'] = False
    epochs, _ = mne.io.set_eeg_reference(epochs)
    epochs.apply_proj()

    # That's all for now
    return epochs
