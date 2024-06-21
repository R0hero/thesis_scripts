from data_selector import data_selector
import georinex as gr
import matplotlib.pyplot as plt
import numpy as np
from sv_functions import calculate_multipath_error, get_azimuth_elevation
import os
from scipy.optimize import curve_fit
import scipy.signal
import re

def get_material(file):
    if 'glass' in file:
        wall_type = 'glass'
    elif 'metal' in file:
        wall_type = 'metal'
    elif 'brick' in file:
        wall_type = 'brick'
    elif 'clear' in file:
        wall_type = 'clear'
    return wall_type

def interpolate_nans(array):
    nan_idx = np.isnan(array)
    non_nan_idx = ~nan_idx
    idx = np.arange(len(array))

    array_interp = np.interp(idx, idx[non_nan_idx], array[non_nan_idx])

    return array_interp

def fit_sin(tt, yy, max_evals=5000):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c):  return A * np.sin(w*t + p) + c
    popt, pcov = curve_fit(sinfunc, tt, yy, p0=guess, maxfev=max_evals)
    A, w, p, c = popt
    f = w/(2.*np.pi)
    fitfunc = lambda t: A * np.sin(w*t + p) + c
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}

def get_date(path):
    expression = r'\d{6}'

    return re.search(expression,path)[0]

def normalize_amplitude_timeseries(data):
    min_val = np.nanmin(data)
    max_val = np.nanmax(data)

    normalized_data = (data - np.nanmean(data))/(max_val - min_val)

    return normalized_data

def normalize_amplitude_timeseries_half(data):
    
    min_val = np.nanmin(data)
    max_val = np.nanmax(data)
    normalized_data = [(-0.5) + ((x - min_val) * (0.5 - (-0.5)) / (max_val - min_val)) for x in data]
    
    return np.array(normalized_data)

def normalize_amplitude_timeseries_whole(dataset):
    min_val = np.nanmin(dataset)
    max_val = np.nanmax(dataset)
    normalized_dataset = [(x - min_val) / (max_val - min_val) * 2 - 1 for x in dataset]
    return np.array(normalized_dataset)

def moving_average(arr, window_size=50):
    padded_arr = np.pad(arr, (window_size // 2, window_size // 2), mode='edge')
    cumsum = np.cumsum(padded_arr)
    moving_avg = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
    return moving_avg

def plot_spectrogram_distance_correlation(sv_list, date, material, obs_short, obs_medium, obs_long, system='gps'):
    for _, sv in enumerate(sv_list):

        if system == 'gps':
            cn0_L1_short = obs_short.sel(sv=sv)[CN0_L1_SELECTOR].values
            cn0_L1_medium = obs_medium.sel(sv=sv)[CN0_L1_SELECTOR].values
            cn0_L1_long = obs_long.sel(sv=sv)[CN0_L1_SELECTOR].values

            cn0_L2_short = obs_short.sel(sv=sv)[CN0_L2_SELECTOR].values
            cn0_L2_medium = obs_medium.sel(sv=sv)[CN0_L2_SELECTOR].values
            cn0_L2_long = obs_long.sel(sv=sv)[CN0_L2_SELECTOR].values

            cn0_L5_short = obs_short.sel(sv=sv)[CN0_L5_SELECTOR].values
            cn0_L5_medium = obs_medium.sel(sv=sv)[CN0_L5_SELECTOR].values
            cn0_L5_long = obs_long.sel(sv=sv)[CN0_L5_SELECTOR].values

            if PERFORM_GENERAL_AMPLITUDE_NORMALIZATION:
                norm_cn0_L1_short = normalize_amplitude_timeseries(cn0_L1_short)
                norm_cn0_L1_medium = normalize_amplitude_timeseries(cn0_L1_medium)
                norm_cn0_L1_long = normalize_amplitude_timeseries(cn0_L1_long)

                norm_cn0_L2_short = normalize_amplitude_timeseries(cn0_L2_short)
                norm_cn0_L2_medium = normalize_amplitude_timeseries(cn0_L2_medium)
                norm_cn0_L2_long = normalize_amplitude_timeseries(cn0_L2_long)

                norm_cn0_L5_short = normalize_amplitude_timeseries(cn0_L5_short)
                norm_cn0_L5_medium = normalize_amplitude_timeseries(cn0_L5_medium)
                norm_cn0_L5_long = normalize_amplitude_timeseries(cn0_L5_long)
            elif PERFORM_HALF_AMPLITUDE_NORMALIZATION:
                norm_cn0_L1_short = normalize_amplitude_timeseries_half(cn0_L1_short)
                norm_cn0_L1_medium = normalize_amplitude_timeseries_half(cn0_L1_medium)
                norm_cn0_L1_long = normalize_amplitude_timeseries_half(cn0_L1_long)

                norm_cn0_L2_short = normalize_amplitude_timeseries_half(cn0_L2_short)
                norm_cn0_L2_medium = normalize_amplitude_timeseries_half(cn0_L2_medium)
                norm_cn0_L2_long = normalize_amplitude_timeseries_half(cn0_L2_long)

                norm_cn0_L5_short = normalize_amplitude_timeseries_half(cn0_L5_short)
                norm_cn0_L5_medium = normalize_amplitude_timeseries_half(cn0_L5_medium)
                norm_cn0_L5_long = normalize_amplitude_timeseries_half(cn0_L5_long)
            elif PERFORM_WHOLE_AMPLITUDE_NORMALIZATION:
                norm_cn0_L1_short = normalize_amplitude_timeseries_whole(cn0_L1_short)
                norm_cn0_L1_medium = normalize_amplitude_timeseries_whole(cn0_L1_medium)
                norm_cn0_L1_long = normalize_amplitude_timeseries_whole(cn0_L1_long)

                norm_cn0_L2_short = normalize_amplitude_timeseries_whole(cn0_L2_short)
                norm_cn0_L2_medium = normalize_amplitude_timeseries_whole(cn0_L2_medium)
                norm_cn0_L2_long = normalize_amplitude_timeseries_whole(cn0_L2_long)

                norm_cn0_L5_short = normalize_amplitude_timeseries_whole(cn0_L5_short)
                norm_cn0_L5_medium = normalize_amplitude_timeseries_whole(cn0_L5_medium)
                norm_cn0_L5_long = normalize_amplitude_timeseries_whole(cn0_L5_long)
            else:
                norm_cn0_L1_short = cn0_L1_short
                norm_cn0_L1_medium = cn0_L1_medium
                norm_cn0_L1_long = cn0_L1_long

                norm_cn0_L2_short = cn0_L2_short
                norm_cn0_L2_medium = cn0_L2_medium
                norm_cn0_L2_long = cn0_L2_long

                norm_cn0_L5_short = cn0_L5_short
                norm_cn0_L5_medium = cn0_L5_medium
                norm_cn0_L5_long = cn0_L5_long

            if PERFORM_MOVING_AVERAGE:
                norm_cn0_L1_short = moving_average(norm_cn0_L1_short,window_size=WINDOW_SIZE)
                norm_cn0_L1_medium = moving_average(norm_cn0_L1_medium,window_size=WINDOW_SIZE)
                norm_cn0_L1_long = moving_average(norm_cn0_L1_long,window_size=WINDOW_SIZE)

                norm_cn0_L2_short = moving_average(norm_cn0_L2_short,window_size=WINDOW_SIZE)
                norm_cn0_L2_medium = moving_average(norm_cn0_L2_medium,window_size=WINDOW_SIZE)
                norm_cn0_L2_long = moving_average(norm_cn0_L2_long,window_size=WINDOW_SIZE)

                norm_cn0_L5_short = moving_average(norm_cn0_L5_short,window_size=WINDOW_SIZE)
                norm_cn0_L5_medium = moving_average(norm_cn0_L5_medium,window_size=WINDOW_SIZE)
                norm_cn0_L5_long = moving_average(norm_cn0_L5_long,window_size=WINDOW_SIZE)

            window = 'blackmanharris'
            nperseg = 1300
            nfft = 2400
            noverlap = 1200
            
            signal_labels = ['L1', 'L2', 'L5']
            cmap = 'viridis'

            f_L1_short, t_L1_short, Sxx_L1_short = scipy.signal.spectrogram(norm_cn0_L1_short, fs = 1, 
                                                                            window = window, nfft = nfft, nperseg = nperseg, noverlap = noverlap)
            f_L1_medium, t_L1_medium, Sxx_L1_medium = scipy.signal.spectrogram(norm_cn0_L1_medium, fs = 1, 
                                                                            window = window, nfft = nfft, nperseg = nperseg, noverlap = noverlap)
            f_L1_long, t_L1_long, Sxx_L1_long = scipy.signal.spectrogram(norm_cn0_L1_long, fs = 1, 
                                                                            window = window, nfft = nfft, nperseg = nperseg, noverlap = noverlap)

            f_L2_short, t_L2_short, Sxx_L2_short = scipy.signal.spectrogram(norm_cn0_L2_short, fs = 1, 
                                                                            window = window, nfft = nfft, nperseg = nperseg, noverlap = noverlap)
            f_L2_medium, t_L2_medium, Sxx_L2_medium = scipy.signal.spectrogram(norm_cn0_L2_medium, fs = 1, 
                                                                            window = window, nfft = nfft, nperseg = nperseg, noverlap = noverlap)
            f_L2_long, t_L2_long, Sxx_L2_long = scipy.signal.spectrogram(norm_cn0_L2_long, fs = 1, 
                                                                            window = window, nfft = nfft, nperseg = nperseg, noverlap = noverlap)

            f_L5_short, t_L5_short, Sxx_L5_short = scipy.signal.spectrogram(norm_cn0_L5_short, fs = 1, 
                                                                            window = window, nfft = nfft, nperseg = nperseg, noverlap = noverlap)
            f_L5_medium, t_L5_medium, Sxx_L5_medium = scipy.signal.spectrogram(norm_cn0_L5_medium, fs = 1, 
                                                                            window = window, nfft = nfft, nperseg = nperseg, noverlap = noverlap)
            f_L5_long, t_L5_long, Sxx_L5_long = scipy.signal.spectrogram(norm_cn0_L5_long, fs = 1, 
                                                                            window = window, nfft = nfft, nperseg = nperseg, noverlap = noverlap)


            fig, ax = plt.subplots(3,3,sharex=True,sharey=True,figsize=(8,9))
            plt.suptitle(f'SV: {sv}')
            im0 = ax[0,0].pcolormesh(t_L1_short, f_L1_short, Sxx_L1_short, shading='nearest',cmap=cmap)
            im1 = ax[1,0].pcolormesh(t_L1_medium, f_L1_medium, Sxx_L1_medium, shading='nearest',cmap=cmap)
            im2 = ax[2,0].pcolormesh(t_L1_long, f_L1_long, Sxx_L1_long, shading='nearest',cmap=cmap)

            im3 = ax[0,1].pcolormesh(t_L2_short, f_L2_short, Sxx_L2_short, shading='nearest',cmap=cmap)
            im4 = ax[1,1].pcolormesh(t_L2_medium, f_L2_medium, Sxx_L2_medium, shading='nearest',cmap=cmap)
            im5 = ax[2,1].pcolormesh(t_L2_long, f_L2_long, Sxx_L2_long, shading='nearest',cmap=cmap)

            im6 = ax[0,2].pcolormesh(t_L5_short, f_L5_short, Sxx_L5_short, shading='nearest',cmap=cmap)
            im7 = ax[1,2].pcolormesh(t_L5_medium, f_L5_medium, Sxx_L5_medium, shading='nearest',cmap=cmap)
            im8 = ax[2,2].pcolormesh(t_L5_long, f_L5_long, Sxx_L5_long, shading='nearest',cmap=cmap)

            for i in range(0,3):
                ax[i,0].set_ylabel('frequency [Hz]')
                ax[2,i].set_xlabel('time [s]')
                ax[0,i].set_title(f'{signal_labels[i]}')

            ax[2,2].set_ylim([0,0.015])

            cbar_ax = fig.add_axes([0.116, -0.01, 0.844, 0.01])  # [left, bottom, width, height]
            cbar = fig.colorbar(im1, cax=cbar_ax, orientation='horizontal')
            cbar.set_label('Intensity')

            # plt.tight_layout()
            plt.savefig(f'../figs/spectrograms/distance_correlation/{material}_{sv}_spectrogram_distances_{date}.png')
            plt.savefig(f'../figs/spectrograms/distance_correlation/{material}_{sv}_spectrogram_distances_{date}.pdf')
        else:
            cn0_E1_short = obs_short.sel(sv=sv)[CN0_E1_SELECTOR].values
            cn0_E1_medium = obs_medium.sel(sv=sv)[CN0_E1_SELECTOR].values
            cn0_E1_long = obs_long.sel(sv=sv)[CN0_E1_SELECTOR].values

            cn0_E5A_short = obs_short.sel(sv=sv)[CN0_E5A_SELECTOR].values
            cn0_E5A_medium = obs_medium.sel(sv=sv)[CN0_E5A_SELECTOR].values
            cn0_E5A_long = obs_long.sel(sv=sv)[CN0_E5A_SELECTOR].values

            cn0_E5B_short = obs_short.sel(sv=sv)[CN0_E5B_SELECTOR].values
            cn0_E5B_medium = obs_medium.sel(sv=sv)[CN0_E5B_SELECTOR].values
            cn0_E5B_long = obs_long.sel(sv=sv)[CN0_E5B_SELECTOR].values

            cn0_E5AB_short = obs_short.sel(sv=sv)[CN0_E5AB_SELECTOR].values
            cn0_E5AB_medium = obs_medium.sel(sv=sv)[CN0_E5AB_SELECTOR].values
            cn0_E5AB_long = obs_long.sel(sv=sv)[CN0_E5AB_SELECTOR].values

            if PERFORM_GENERAL_AMPLITUDE_NORMALIZATION:
                norm_cn0_E1_short = normalize_amplitude_timeseries(cn0_E1_short)
                norm_cn0_E1_medium = normalize_amplitude_timeseries(cn0_E1_medium)
                norm_cn0_E1_long = normalize_amplitude_timeseries(cn0_E1_long)

                norm_cn0_E5A_short = normalize_amplitude_timeseries(cn0_E5A_short)
                norm_cn0_E5A_medium = normalize_amplitude_timeseries(cn0_E5A_medium)
                norm_cn0_E5A_long = normalize_amplitude_timeseries(cn0_E5A_long)

                norm_cn0_E5B_short = normalize_amplitude_timeseries(cn0_E5B_short)
                norm_cn0_E5B_medium = normalize_amplitude_timeseries(cn0_E5B_medium)
                norm_cn0_E5B_long = normalize_amplitude_timeseries(cn0_E5B_long)

                norm_cn0_E5AB_short = normalize_amplitude_timeseries(cn0_E5AB_short)
                norm_cn0_E5AB_medium = normalize_amplitude_timeseries(cn0_E5AB_medium)
                norm_cn0_E5AB_long = normalize_amplitude_timeseries(cn0_E5AB_long)
            elif PERFORM_HALF_AMPLITUDE_NORMALIZATION:
                norm_cn0_E1_short = normalize_amplitude_timeseries_half(cn0_E1_short)
                norm_cn0_E1_medium = normalize_amplitude_timeseries_half(cn0_E1_medium)
                norm_cn0_E1_long = normalize_amplitude_timeseries_half(cn0_E1_long)

                norm_cn0_E5A_short = normalize_amplitude_timeseries_half(cn0_E5A_short)
                norm_cn0_E5A_medium = normalize_amplitude_timeseries_half(cn0_E5A_medium)
                norm_cn0_E5A_long = normalize_amplitude_timeseries_half(cn0_E5A_long)

                norm_cn0_E5B_short = normalize_amplitude_timeseries_half(cn0_E5B_short)
                norm_cn0_E5B_medium = normalize_amplitude_timeseries_half(cn0_E5B_medium)
                norm_cn0_E5B_long = normalize_amplitude_timeseries_half(cn0_E5B_long)

                norm_cn0_E5AB_short = normalize_amplitude_timeseries_half(cn0_E5AB_short)
                norm_cn0_E5AB_medium = normalize_amplitude_timeseries_half(cn0_E5AB_medium)
                norm_cn0_E5AB_long = normalize_amplitude_timeseries_half(cn0_E5AB_long)
            elif PERFORM_WHOLE_AMPLITUDE_NORMALIZATION:
                norm_cn0_E1_short = normalize_amplitude_timeseries_whole(cn0_E1_short)
                norm_cn0_E1_medium = normalize_amplitude_timeseries_whole(cn0_E1_medium)
                norm_cn0_E1_long = normalize_amplitude_timeseries_whole(cn0_E1_long)

                norm_cn0_E5A_short = normalize_amplitude_timeseries_whole(cn0_E5A_short)
                norm_cn0_E5A_medium = normalize_amplitude_timeseries_whole(cn0_E5A_medium)
                norm_cn0_E5A_long = normalize_amplitude_timeseries_whole(cn0_E5A_long)

                norm_cn0_E5B_short = normalize_amplitude_timeseries_whole(cn0_E5B_short)
                norm_cn0_E5B_medium = normalize_amplitude_timeseries_whole(cn0_E5B_medium)
                norm_cn0_E5B_long = normalize_amplitude_timeseries_whole(cn0_E5B_long)

                norm_cn0_E5AB_short = normalize_amplitude_timeseries_whole(cn0_E5AB_short)
                norm_cn0_E5AB_medium = normalize_amplitude_timeseries_whole(cn0_E5AB_medium)
                norm_cn0_E5AB_long = normalize_amplitude_timeseries_whole(cn0_E5AB_long)
            else:
                norm_cn0_E1_short = cn0_E1_short
                norm_cn0_E1_medium = cn0_E1_medium
                norm_cn0_E1_long = cn0_E1_long

                norm_cn0_E5A_short = cn0_E5A_short
                norm_cn0_E5A_medium = cn0_E5A_medium
                norm_cn0_E5A_long = cn0_E5A_long

                norm_cn0_E5B_short = cn0_E5B_short
                norm_cn0_E5B_medium = cn0_E5B_medium
                norm_cn0_E5B_long = cn0_E5B_long

                norm_cn0_E5AB_short = cn0_E5AB_short
                norm_cn0_E5AB_medium = cn0_E5AB_medium
                norm_cn0_E5AB_long = cn0_E5AB_long

            if PERFORM_MOVING_AVERAGE:
                norm_cn0_E1_short = moving_average(norm_cn0_E1_short,window_size=WINDOW_SIZE)
                norm_cn0_E1_medium = moving_average(norm_cn0_E1_medium,window_size=WINDOW_SIZE)
                norm_cn0_E1_long = moving_average(norm_cn0_E1_long,window_size=WINDOW_SIZE)

                norm_cn0_E5A_short = moving_average(norm_cn0_E5A_short,window_size=WINDOW_SIZE)
                norm_cn0_E5A_medium = moving_average(norm_cn0_E5A_medium,window_size=WINDOW_SIZE)
                norm_cn0_E5A_long = moving_average(norm_cn0_E5A_long,window_size=WINDOW_SIZE)

                norm_cn0_E5B_short = moving_average(norm_cn0_E5B_short,window_size=WINDOW_SIZE)
                norm_cn0_E5B_medium = moving_average(norm_cn0_E5B_medium,window_size=WINDOW_SIZE)
                norm_cn0_E5B_long = moving_average(norm_cn0_E5B_long,window_size=WINDOW_SIZE)

                norm_cn0_E5AB_short = moving_average(norm_cn0_E5AB_short,window_size=WINDOW_SIZE)
                norm_cn0_E5AB_medium = moving_average(norm_cn0_E5AB_medium,window_size=WINDOW_SIZE)
                norm_cn0_E5AB_long = moving_average(norm_cn0_E5AB_long,window_size=WINDOW_SIZE)

            window = 'blackmanharris'
            nperseg = 1300
            nfft = 2400
            noverlap = 1200
            
            signal_labels = ['E1','E5A','E5B','E5AB']
            cmap = 'viridis'

            f_E1_short, t_E1_short, Sxx_E1_short = scipy.signal.spectrogram(norm_cn0_E1_short, fs = 1, 
                                                                            window = window, nfft = nfft, nperseg = nperseg, noverlap = noverlap)
            f_E1_medium, t_E1_medium, Sxx_E1_medium = scipy.signal.spectrogram(norm_cn0_E1_medium, fs = 1, 
                                                                            window = window, nfft = nfft, nperseg = nperseg, noverlap = noverlap)
            f_E1_long, t_E1_long, Sxx_E1_long = scipy.signal.spectrogram(norm_cn0_E1_long, fs = 1, 
                                                                            window = window, nfft = nfft, nperseg = nperseg, noverlap = noverlap)

            f_E5A_short, t_E5A_short, Sxx_E5A_short = scipy.signal.spectrogram(norm_cn0_E5A_short, fs = 1, 
                                                                            window = window, nfft = nfft, nperseg = nperseg, noverlap = noverlap)
            f_E5A_medium, t_E5A_medium, Sxx_E5A_medium = scipy.signal.spectrogram(norm_cn0_E5A_medium, fs = 1, 
                                                                            window = window, nfft = nfft, nperseg = nperseg, noverlap = noverlap)
            f_E5A_long, t_E5A_long, Sxx_E5A_long = scipy.signal.spectrogram(norm_cn0_E5A_long, fs = 1, 
                                                                            window = window, nfft = nfft, nperseg = nperseg, noverlap = noverlap)

            f_E5B_short, t_E5B_short, Sxx_E5B_short = scipy.signal.spectrogram(norm_cn0_E5B_short, fs = 1, 
                                                                            window = window, nfft = nfft, nperseg = nperseg, noverlap = noverlap)
            f_E5B_medium, t_E5B_medium, Sxx_E5B_medium = scipy.signal.spectrogram(norm_cn0_E5B_medium, fs = 1, 
                                                                            window = window, nfft = nfft, nperseg = nperseg, noverlap = noverlap)
            f_E5B_long, t_E5B_long, Sxx_E5B_long = scipy.signal.spectrogram(norm_cn0_E5B_long, fs = 1, 
                                                                            window = window, nfft = nfft, nperseg = nperseg, noverlap = noverlap)

            f_E5AB_short, t_E5AB_short, Sxx_E5AB_short = scipy.signal.spectrogram(norm_cn0_E5AB_short, fs = 1, 
                                                                            window = window, nfft = nfft, nperseg = nperseg, noverlap = noverlap)
            f_E5AB_medium, t_E5AB_medium, Sxx_E5AB_medium = scipy.signal.spectrogram(norm_cn0_E5AB_medium, fs = 1, 
                                                                            window = window, nfft = nfft, nperseg = nperseg, noverlap = noverlap)
            f_E5AB_long, t_E5AB_long, Sxx_E5AB_long = scipy.signal.spectrogram(norm_cn0_E5AB_long, fs = 1, 
                                                                            window = window, nfft = nfft, nperseg = nperseg, noverlap = noverlap)

            fig, ax = plt.subplots(3,4,sharex=True,sharey=True,figsize=(8,9))
            plt.suptitle(f'SV: {sv}')
            im0 = ax[0,0].pcolormesh(t_E1_short, f_E1_short, Sxx_E1_short, shading='nearest',cmap=cmap)
            im1 = ax[1,0].pcolormesh(t_E1_medium, f_E1_medium, Sxx_E1_medium, shading='nearest',cmap=cmap)
            im2 = ax[2,0].pcolormesh(t_E1_long, f_E1_long, Sxx_E1_long, shading='nearest',cmap=cmap)

            im3 = ax[0,1].pcolormesh(t_E5A_short, f_E5A_short, Sxx_E5A_short, shading='nearest',cmap=cmap)
            im4 = ax[1,1].pcolormesh(t_E5A_medium, f_E5A_medium, Sxx_E5A_medium, shading='nearest',cmap=cmap)
            im5 = ax[2,1].pcolormesh(t_E5A_long, f_E5A_long, Sxx_E5A_long, shading='nearest',cmap=cmap)

            im6 = ax[0,2].pcolormesh(t_E5B_short, f_E5B_short, Sxx_E5B_short, shading='nearest',cmap=cmap)
            im7 = ax[1,2].pcolormesh(t_E5B_medium, f_E5B_medium, Sxx_E5B_medium, shading='nearest',cmap=cmap)
            im8 = ax[2,2].pcolormesh(t_E5B_long, f_E5B_long, Sxx_E5B_long, shading='nearest',cmap=cmap)

            im9 = ax[0,3].pcolormesh(t_E5AB_short, f_E5AB_short, Sxx_E5AB_short, shading='nearest',cmap=cmap)
            im10 = ax[1,3].pcolormesh(t_E5AB_medium, f_E5AB_medium, Sxx_E5AB_medium, shading='nearest',cmap=cmap)
            im11 = ax[2,3].pcolormesh(t_E5AB_long, f_E5AB_long, Sxx_E5AB_long, shading='nearest',cmap=cmap)

            for i in range(0,4):
                ax[2,i].set_xlabel('time [s]')
                ax[0,i].set_title(f'{signal_labels[i]}')
            for i in range(0,3):
                ax[i,0].set_ylabel('frequency [Hz]')

            ax[2,2].set_ylim([0,0.015])

            cbar_ax = fig.add_axes([0.116, -0.01, 0.844, 0.01])  # [left, bottom, width, height]
            cbar = fig.colorbar(im1, cax=cbar_ax, orientation='horizontal')
            cbar.set_label('Intensity')

            # plt.tight_layout()
            plt.savefig(f'../figs/spectrograms/distance_correlation/{material}_{sv}_spectrogram_distances_{date}.png')
            plt.savefig(f'../figs/spectrograms/distance_correlation/{material}_{sv}_spectrogram_distances_{date}.pdf')

def plot_spectrogram_RHPvLHP(sv_list, date, material, obs_main, obs_aux, system='gps'):

    for sv in sv_list:
        print(sv)
        if system == 'gps':
            cn0_L1_main = obs_main.sel(sv=sv)[CN0_L1_SELECTOR].values
            cn0_L2_main = obs_main.sel(sv=sv)[CN0_L2_SELECTOR].values
            cn0_L5_main = obs_main.sel(sv=sv)[CN0_L5_SELECTOR].values

            cn0_L1_aux = obs_aux.sel(sv=sv)[CN0_L1_SELECTOR].values
            cn0_L2_aux = obs_aux.sel(sv=sv)[CN0_L2_SELECTOR].values
            cn0_L5_aux = obs_aux.sel(sv=sv)[CN0_L5_SELECTOR].values

            if np.sum(np.isnan(cn0_L1_main)) > 0 and np.sum(np.isnan(cn0_L1_main)) < len(cn0_L1_main):
                cn0_L1_main = interpolate_nans(cn0_L1_main)
            if np.sum(np.isnan(cn0_L1_aux)) > 0 and np.sum(np.isnan(cn0_L1_aux)) < len(cn0_L1_aux):
                cn0_L1_aux = interpolate_nans(cn0_L1_aux)
            
            if np.sum(np.isnan(cn0_L2_main)) > 0 and np.sum(np.isnan(cn0_L2_main)) < len(cn0_L2_main):
                cn0_L2_main = interpolate_nans(cn0_L2_main)
            if np.sum(np.isnan(cn0_L2_aux)) > 0 and np.sum(np.isnan(cn0_L2_aux)) < len(cn0_L2_aux):
                cn0_L2_aux = interpolate_nans(cn0_L2_aux)

            if np.sum(np.isnan(cn0_L5_main)) > 0 and np.sum(np.isnan(cn0_L5_main)) < len(cn0_L5_main):
                cn0_L5_main = interpolate_nans(cn0_L5_main)
            if np.sum(np.isnan(cn0_L5_aux)) > 0 and np.sum(np.isnan(cn0_L5_aux)) < len(cn0_L5_aux):
                cn0_L5_aux = interpolate_nans(cn0_L5_aux)

            if PERFORM_GENERAL_AMPLITUDE_NORMALIZATION:
                cn0_L1_main_norm = normalize_amplitude_timeseries(cn0_L1_main)
                cn0_L2_main_norm = normalize_amplitude_timeseries(cn0_L2_main)
                cn0_L5_main_norm = normalize_amplitude_timeseries(cn0_L5_main)
                cn0_L1_aux_norm = normalize_amplitude_timeseries(cn0_L1_aux)
                cn0_L2_aux_norm = normalize_amplitude_timeseries(cn0_L2_aux)
                cn0_L5_aux_norm = normalize_amplitude_timeseries(cn0_L5_aux)
            elif PERFORM_HALF_AMPLITUDE_NORMALIZATION:
                cn0_L1_main_norm = normalize_amplitude_timeseries_half(cn0_L1_main)
                cn0_L2_main_norm = normalize_amplitude_timeseries_half(cn0_L2_main)
                cn0_L5_main_norm = normalize_amplitude_timeseries_half(cn0_L5_main)
                cn0_L1_aux_norm = normalize_amplitude_timeseries_half(cn0_L1_aux)
                cn0_L2_aux_norm = normalize_amplitude_timeseries_half(cn0_L2_aux)
                cn0_L5_aux_norm = normalize_amplitude_timeseries_half(cn0_L5_aux)
            elif PERFORM_WHOLE_AMPLITUDE_NORMALIZATION:
                cn0_L1_main_norm = normalize_amplitude_timeseries_whole(cn0_L1_main)
                cn0_L2_main_norm = normalize_amplitude_timeseries_whole(cn0_L2_main)
                cn0_L5_main_norm = normalize_amplitude_timeseries_whole(cn0_L5_main)
                cn0_L1_aux_norm = normalize_amplitude_timeseries_whole(cn0_L1_aux)
                cn0_L2_aux_norm = normalize_amplitude_timeseries_whole(cn0_L2_aux)
                cn0_L5_aux_norm = normalize_amplitude_timeseries_whole(cn0_L5_aux)
            else:
                cn0_L1_main_norm = cn0_L1_main
                cn0_L2_main_norm = cn0_L2_main
                cn0_L5_main_norm = cn0_L5_main
                cn0_L1_aux_norm = cn0_L1_aux
                cn0_L2_aux_norm = cn0_L2_aux
                cn0_L5_aux_norm = cn0_L5_aux

            if PERFORM_MOVING_AVERAGE:
                cn0_L1_main_norm = moving_average(cn0_L1_main_norm,window_size=WINDOW_SIZE)
                cn0_L2_main_norm = moving_average(cn0_L2_main_norm,window_size=WINDOW_SIZE)
                cn0_L5_main_norm = moving_average(cn0_L5_main_norm,window_size=WINDOW_SIZE)
                cn0_L1_aux_norm = moving_average(cn0_L1_aux_norm,window_size=WINDOW_SIZE)
                cn0_L2_aux_norm = moving_average(cn0_L2_aux_norm,window_size=WINDOW_SIZE)
                cn0_L5_aux_norm = moving_average(cn0_L5_aux_norm,window_size=WINDOW_SIZE)

            window = 'blackmanharris'
            nperseg = 1300
            nfft = 2400
            noverlap = 1200
            spectrum='density'
            mode='magnitude'
            shading='gouraud'

            signal_labels = ['RHP', 'LHP']
            cmap = 'viridis'

            f_L1_main, t_L1_main, Sxx_L1_main = scipy.signal.spectrogram(cn0_L1_main_norm, fs = 1, 
                                                                            window = window, nfft = nfft, nperseg = nperseg, noverlap = noverlap,
                                                                            scaling = spectrum, mode = mode)
            f_L1_aux, t_L1_aux, Sxx_L1_aux = scipy.signal.spectrogram(cn0_L1_aux_norm, fs = 1, 
                                                                            window = window, nfft = nfft, nperseg = nperseg, noverlap = noverlap,
                                                                            scaling = spectrum, mode = mode)

            f_L2_main, t_L2_main, Sxx_L2_main = scipy.signal.spectrogram(cn0_L2_main_norm, fs = 1, 
                                                                            window = window, nfft = nfft, nperseg = nperseg, noverlap = noverlap,
                                                                            scaling = spectrum, mode = mode)
            f_L2_aux, t_L2_aux, Sxx_L2_aux = scipy.signal.spectrogram(cn0_L2_aux_norm, fs = 1, 
                                                                            window = window, nfft = nfft, nperseg = nperseg, noverlap = noverlap,
                                                                            scaling = spectrum, mode = mode)

            f_L5_main, t_L5_main, Sxx_L5_main = scipy.signal.spectrogram(cn0_L5_main_norm, fs = 1, 
                                                                            window = window, nfft = nfft, nperseg = nperseg, noverlap = noverlap,
                                                                            scaling = spectrum, mode = mode)
            f_L5_aux, t_L5_aux, Sxx_L5_aux = scipy.signal.spectrogram(cn0_L5_aux_norm, fs = 1, 
                                                                            window = window, nfft = nfft, nperseg = nperseg, noverlap = noverlap,
                                                                            scaling = spectrum, mode = mode)

            vmin = 0 #min(np.min(Sxx_L1_main), np.min(Sxx_L1_aux), np.min(Sxx_L2_main), np.min(Sxx_L2_aux), np.min(Sxx_L5_main), np.min(Sxx_L5_aux))
            vmax = max(np.max(Sxx_L1_main), np.max(Sxx_L1_aux), np.max(Sxx_L2_main), np.max(Sxx_L2_aux), np.max(Sxx_L5_main), np.max(Sxx_L5_aux))
            
            f_L1_main *= 1000
            f_L1_aux *= 1000
            f_L2_main *= 1000
            f_L2_aux *= 1000
            f_L5_main *= 1000
            f_L5_aux *= 1000

            fig, ax = plt.subplots(3,2,sharex=True,sharey=True,figsize=(8,9))
            plt.suptitle(f'SV: {sv}')
            im0 = ax[0,0].pcolormesh(t_L1_main, f_L1_main, Sxx_L1_main, shading=shading, cmap=cmap, vmin=vmin, vmax=vmax)
            im1 = ax[0,1].pcolormesh(t_L1_aux, f_L1_aux, Sxx_L1_aux, shading=shading, cmap=cmap, vmin=vmin, vmax=vmax)

            im2 = ax[1,0].pcolormesh(t_L2_main, f_L2_main, Sxx_L2_main, shading=shading, cmap=cmap, vmin=vmin, vmax=vmax)
            im3 = ax[1,1].pcolormesh(t_L2_aux, f_L2_aux, Sxx_L2_aux, shading=shading, cmap=cmap, vmin=vmin, vmax=vmax)

            im4 = ax[2,0].pcolormesh(t_L5_main, f_L5_main, Sxx_L5_main, shading=shading, cmap=cmap, vmin=vmin, vmax=vmax)
            im5 = ax[2,1].pcolormesh(t_L5_aux, f_L5_aux, Sxx_L5_aux, shading=shading, cmap=cmap, vmin=vmin, vmax=vmax)

            for i in range(0,2):
                ax[2,i].set_xlabel('Time [s]')
                ax[0,i].set_title(f'{signal_labels[i]}',fontsize=14, pad=20)
                ax[0,i].text(0.5, 1.02, 'L1', transform=ax[0, i].transAxes, fontsize=11, va='bottom', ha='center')
                ax[1,i].text(0.5, 1.02, 'L2', transform=ax[1, i].transAxes, fontsize=11, va='bottom', ha='center')
                ax[2,i].text(0.5, 1.02, 'L5', transform=ax[2, i].transAxes, fontsize=11, va='bottom', ha='center')
            for i in range(0,3):
                ax[i,0].set_ylabel('Frequency [mHz]')

            if material == 'clear':
                ax[2,1].set_ylim([0,30])
            else:
                ax[2,1].set_ylim([0,15])

            cbar = fig.colorbar(im0, ax=ax.ravel().tolist(), orientation='vertical', aspect=50, pad=0.05)
            cbar.set_label('Magnitude [dB]')

            plt.savefig(f'../figs/spectrograms/RHPvLHP/{material}_{sv}_spectrogram_RHPvLHP_{date}.png')
            plt.savefig(f'../figs/spectrograms/RHPvLHP/{material}_{sv}_spectrogram_RHPvLHP_{date}.pdf',dpi=300,bbox_inches='tight')
        else:
            cn0_E1_main = obs_main.sel(sv=sv)[CN0_E1_SELECTOR].values
            cn0_E5a_main = obs_main.sel(sv=sv)[CN0_E5A_SELECTOR].values
            cn0_E5b_main = obs_main.sel(sv=sv)[CN0_E5B_SELECTOR].values
            cn0_E5ab_main = obs_main.sel(sv=sv)[CN0_E5AB_SELECTOR].values

            cn0_E1_aux = obs_aux.sel(sv=sv)[CN0_E1_SELECTOR].values
            cn0_E5a_aux = obs_aux.sel(sv=sv)[CN0_E5A_SELECTOR].values
            cn0_E5b_aux = obs_aux.sel(sv=sv)[CN0_E5B_SELECTOR].values
            cn0_E5ab_aux = obs_aux.sel(sv=sv)[CN0_E5AB_SELECTOR].values

            if np.sum(np.isnan(cn0_E1_main)) > 0 and np.sum(np.isnan(cn0_E1_main)) < len(cn0_E1_main):
                cn0_E1_main = interpolate_nans(cn0_E1_main)
            if np.sum(np.isnan(cn0_E1_aux)) > 0 and np.sum(np.isnan(cn0_E1_aux)) < len(cn0_E1_aux):
                cn0_E1_aux = interpolate_nans(cn0_E1_aux)

            if np.sum(np.isnan(cn0_E5a_main)) > 0 and np.sum(np.isnan(cn0_E5a_main)) < len(cn0_E5a_main):
                cn0_E5a_main = interpolate_nans(cn0_E5a_main)
            if np.sum(np.isnan(cn0_E5a_aux)) > 0 and np.sum(np.isnan(cn0_E5a_aux)) < len(cn0_E5a_aux):
                cn0_E5a_aux = interpolate_nans(cn0_E5a_aux)

            if np.sum(np.isnan(cn0_E5b_main)) > 0 and np.sum(np.isnan(cn0_E5b_main)) < len(cn0_E5b_main):
                cn0_E5b_main = interpolate_nans(cn0_E5b_main)
            if np.sum(np.isnan(cn0_E5b_aux)) > 0 and np.sum(np.isnan(cn0_E5b_aux)) < len(cn0_E5b_aux):
                cn0_E5b_aux = interpolate_nans(cn0_E5b_aux)

            if np.sum(np.isnan(cn0_E5ab_main)) > 0 and np.sum(np.isnan(cn0_E5ab_main)) < len(cn0_E5ab_main):
                cn0_E5ab_main = interpolate_nans(cn0_E5ab_main)
            if np.sum(np.isnan(cn0_E5ab_aux)) > 0 and np.sum(np.isnan(cn0_E5ab_aux)) < len(cn0_E5ab_aux):
                cn0_E5ab_aux = interpolate_nans(cn0_E5ab_aux)

            if PERFORM_GENERAL_AMPLITUDE_NORMALIZATION:
                cn0_E1_main_norm = normalize_amplitude_timeseries(cn0_E1_main)
                cn0_E5a_main_norm = normalize_amplitude_timeseries(cn0_E5a_main)
                cn0_E5b_main_norm = normalize_amplitude_timeseries(cn0_E5b_main)
                cn0_E5ab_main_norm = normalize_amplitude_timeseries(cn0_E5ab_main)
                cn0_E1_aux_norm = normalize_amplitude_timeseries(cn0_E1_aux)
                cn0_E5a_aux_norm = normalize_amplitude_timeseries(cn0_E5a_aux)
                cn0_E5b_aux_norm = normalize_amplitude_timeseries(cn0_E5b_aux)
                cn0_E5ab_aux_norm = normalize_amplitude_timeseries(cn0_E5ab_aux)
            elif PERFORM_HALF_AMPLITUDE_NORMALIZATION:
                cn0_E1_main_norm = normalize_amplitude_timeseries_half(cn0_E1_main)
                cn0_E5a_main_norm = normalize_amplitude_timeseries_half(cn0_E5a_main)
                cn0_E5b_main_norm = normalize_amplitude_timeseries_half(cn0_E5b_main)
                cn0_E5ab_main_norm = normalize_amplitude_timeseries_half(cn0_E5ab_main)
                cn0_E1_aux_norm = normalize_amplitude_timeseries_half(cn0_E1_aux)
                cn0_E5a_aux_norm = normalize_amplitude_timeseries_half(cn0_E5a_aux)
                cn0_E5b_aux_norm = normalize_amplitude_timeseries_half(cn0_E5b_aux)
                cn0_E5ab_aux_norm = normalize_amplitude_timeseries_half(cn0_E5ab_aux)
            elif PERFORM_WHOLE_AMPLITUDE_NORMALIZATION:
                cn0_E1_main_norm = normalize_amplitude_timeseries_whole(cn0_E1_main)
                cn0_E5a_main_norm = normalize_amplitude_timeseries_whole(cn0_E5a_main)
                cn0_E5b_main_norm = normalize_amplitude_timeseries_whole(cn0_E5b_main)
                cn0_E5ab_main_norm = normalize_amplitude_timeseries_whole(cn0_E5ab_main)
                cn0_E1_aux_norm = normalize_amplitude_timeseries_whole(cn0_E1_aux)
                cn0_E5a_aux_norm = normalize_amplitude_timeseries_whole(cn0_E5a_aux)
                cn0_E5b_aux_norm = normalize_amplitude_timeseries_whole(cn0_E5b_aux)
                cn0_E5ab_aux_norm = normalize_amplitude_timeseries_whole(cn0_E5ab_aux)
            else:
                cn0_E1_main_norm = cn0_E1_main
                cn0_E5a_main_norm = cn0_E5a_main
                cn0_E5b_main_norm = cn0_E5b_main
                cn0_E5ab_main_norm = cn0_E5ab_main
                cn0_E1_aux_norm = cn0_E1_aux
                cn0_E5a_aux_norm = cn0_E5a_aux
                cn0_E5b_aux_norm = cn0_E5b_aux
                cn0_E5ab_aux_norm = cn0_E5ab_aux

            if PERFORM_MOVING_AVERAGE:
                cn0_E1_main_norm = moving_average(cn0_E1_main_norm,window_size=WINDOW_SIZE)
                cn0_E5a_main_norm = moving_average(cn0_E5a_main_norm,window_size=WINDOW_SIZE)
                cn0_E5b_main_norm = moving_average(cn0_E5b_main_norm,window_size=WINDOW_SIZE)
                cn0_E5ab_main_norm = moving_average(cn0_E5ab_main_norm,window_size=WINDOW_SIZE)
                cn0_E1_aux_norm = moving_average(cn0_E1_aux_norm,window_size=WINDOW_SIZE)
                cn0_E5a_aux_norm = moving_average(cn0_E5a_aux_norm,window_size=WINDOW_SIZE)
                cn0_E5b_aux_norm = moving_average(cn0_E5b_aux_norm,window_size=WINDOW_SIZE)
                cn0_E5ab_aux_norm = moving_average(cn0_E5ab_aux_norm,window_size=WINDOW_SIZE)

            window = 'blackmanharris'
            nperseg = 1300
            nfft = 2400
            noverlap = 1200
            spectrum='density'
            mode='magnitude'
            shading='gouraud'
            
            signal_labels = ['RHP', 'LHP']
            cmap = 'viridis'

            f_E1_main, t_E1_main, Sxx_E1_main = scipy.signal.spectrogram(cn0_E1_main_norm, fs = 1, 
                                                                            window = window, nfft = nfft, nperseg = nperseg, noverlap = noverlap,
                                                                            scaling=spectrum, mode=mode)
            f_E1_aux, t_E1_aux, Sxx_E1_aux = scipy.signal.spectrogram(cn0_E1_aux_norm, fs = 1, 
                                                                            window = window, nfft = nfft, nperseg = nperseg, noverlap = noverlap,
                                                                            scaling=spectrum, mode=mode)

            f_E5a_main, t_E5a_main, Sxx_E5a_main = scipy.signal.spectrogram(cn0_E5a_main_norm, fs = 1, 
                                                                            window = window, nfft = nfft, nperseg = nperseg, noverlap = noverlap,
                                                                            scaling=spectrum, mode=mode)
            f_E5a_aux, t_E5a_aux, Sxx_E5a_aux = scipy.signal.spectrogram(cn0_E5a_aux_norm, fs = 1, 
                                                                            window = window, nfft = nfft, nperseg = nperseg, noverlap = noverlap,
                                                                            scaling=spectrum, mode=mode)

            f_E5b_main, t_E5b_main, Sxx_E5b_main = scipy.signal.spectrogram(cn0_E5b_main_norm, fs = 1, 
                                                                            window = window, nfft = nfft, nperseg = nperseg, noverlap = noverlap,
                                                                            scaling=spectrum, mode=mode)
            f_E5b_aux, t_E5b_aux, Sxx_E5b_aux = scipy.signal.spectrogram(cn0_E5b_aux_norm, fs = 1, 
                                                                            window = window, nfft = nfft, nperseg = nperseg, noverlap = noverlap,
                                                                            scaling=spectrum, mode=mode)

            f_E5ab_main, t_E5ab_main, Sxx_E5ab_main = scipy.signal.spectrogram(cn0_E5ab_main_norm, fs = 1, 
                                                                            window = window, nfft = nfft, nperseg = nperseg, noverlap = noverlap,
                                                                            scaling=spectrum, mode=mode)
            f_E5ab_aux, t_E5ab_aux, Sxx_E5ab_aux = scipy.signal.spectrogram(cn0_E5ab_aux_norm, fs = 1, 
                                                                            window = window, nfft = nfft, nperseg = nperseg, noverlap = noverlap,
                                                                            scaling=spectrum, mode=mode)

            f_E1_main *= 1000
            f_E1_aux *= 1000
            f_E5a_main *= 1000
            f_E5a_aux *= 1000
            f_E5b_main *= 1000
            f_E5b_aux *= 1000
            f_E5ab_main *= 1000
            f_E5ab_aux *= 1000
            
            vmin = 0 #min(np.max(Sxx_E1_main), np.max(Sxx_E1_aux), np.max(Sxx_E5a_main), np.max(Sxx_E5a_aux), np.max(Sxx_E5b_main), np.max(Sxx_E5b_aux), np.max(Sxx_E5ab_main), np.max(Sxx_E5ab_aux))
            vmax = max(np.max(Sxx_E1_main), np.max(Sxx_E1_aux), np.max(Sxx_E5a_main), np.max(Sxx_E5a_aux), np.max(Sxx_E5b_main), np.max(Sxx_E5b_aux), np.max(Sxx_E5ab_main), np.max(Sxx_E5ab_aux))

            fig, ax = plt.subplots(4,2,sharex=True,sharey=True,figsize=(8,9))
            plt.suptitle(f'SV: {sv}')
            im0 = ax[0,0].pcolormesh(t_E1_main, f_E1_main, Sxx_E1_main, shading=shading, cmap=cmap, vmin=vmin, vmax=vmax)
            im1 = ax[0,1].pcolormesh(t_E1_aux, f_E1_aux, Sxx_E1_aux, shading=shading, cmap=cmap, vmin=vmin, vmax=vmax)

            im2 = ax[1,0].pcolormesh(t_E5a_main, f_E5a_main, Sxx_E5a_main, shading=shading, cmap=cmap, vmin=vmin, vmax=vmax)
            im3 = ax[1,1].pcolormesh(t_E5a_aux, f_E5a_aux, Sxx_E5a_aux, shading=shading, cmap=cmap, vmin=vmin, vmax=vmax)

            im4 = ax[2,0].pcolormesh(t_E5b_main, f_E5b_main, Sxx_E5b_main, shading=shading, cmap=cmap, vmin=vmin, vmax=vmax)
            im5 = ax[2,1].pcolormesh(t_E5b_aux, f_E5b_aux, Sxx_E5b_aux, shading=shading, cmap=cmap, vmin=vmin, vmax=vmax)

            im4 = ax[3,0].pcolormesh(t_E5ab_main, f_E5ab_main, Sxx_E5ab_main, shading=shading, cmap=cmap, vmin=vmin, vmax=vmax)
            im5 = ax[3,1].pcolormesh(t_E5ab_aux, f_E5ab_aux, Sxx_E5ab_aux, shading=shading, cmap=cmap, vmin=vmin, vmax=vmax)

            for i in range(0,2):
                ax[3,i].set_xlabel('Time [s]')
                ax[0,i].set_title(f'{signal_labels[i]}', fontsize=14, pad=20)
                ax[0,i].text(0.5, 1.02, 'E1', transform=ax[0, i].transAxes, fontsize=11, va='bottom', ha='center')
                ax[1,i].text(0.5, 1.02, 'E5a', transform=ax[1, i].transAxes, fontsize=11, va='bottom', ha='center')
                ax[2,i].text(0.5, 1.02, 'E5b', transform=ax[2, i].transAxes, fontsize=11, va='bottom', ha='center')
                ax[3,i].text(0.5, 1.02, 'E5ab', transform=ax[3, i].transAxes, fontsize=11, va='bottom', ha='center')
            for i in range(0,4):
                ax[i,0].set_ylabel('Frequency [mHz]')
            if material == 'clear':
                ax[2,1].set_ylim([0,30])
            else:
                ax[2,1].set_ylim([0,15])

            print('RHP')
            print(np.max(np.max(Sxx_E1_main)))
            print(np.max(np.max(Sxx_E5a_main)))
            print(np.max(np.max(Sxx_E5b_main)))
            print(np.max(np.max(Sxx_E5ab_main)))
            print('LHP')
            print(np.max(np.max(Sxx_E1_aux)))
            print(np.max(np.max(Sxx_E5a_aux)))
            print(np.max(np.max(Sxx_E5b_aux)))
            print(np.max(np.max(Sxx_E5ab_aux)))
            
            cbar = fig.colorbar(im0, ax=ax.ravel().tolist(), orientation='vertical', aspect=50, pad=0.05)
            cbar.set_label('Magnitude [dB]')

            plt.savefig(f'../figs/spectrograms/RHPvLHP/{material}_{sv}_spectrogram_RHPvLHP_{date}.png')
            plt.savefig(f'../figs/spectrograms/RHPvLHP/{material}_{sv}_spectrogram_RHPvLHP_{date}.pdf',dpi=300, bbox_inches='tight')

def main():
    files = data_selector()

    if len(files) == 6:
        obs_short, nav_short, obs_medium, nav_medium, obs_long, nav_long = files
        date = get_date(obs_short)
        material = get_material(obs_short)
        if 'galileo' in nav_short:
            obs_short = gr.load(obs_short,use="E")
            obs_medium = gr.load(obs_medium,use="E")
            obs_long = gr.load(obs_long,use="E")
            nav_short = gr.load(nav_short)
            nav_medium = gr.load(nav_medium)
            nav_long = gr.load(nav_long)
            
            sv_list_gal_short = obs_short.sv.values
            sv_list_gal_medium = obs_medium.sv.values
            sv_list_gal_long = obs_long.sv.values
            sv_list = [sv for sv in sv_list_gal_long if sv in sv_list_gal_medium and sv in sv_list_gal_short]
            system = 'galileo'
        else:
            obs_short = gr.load(obs_short,use="G")
            obs_medium = gr.load(obs_medium,use="G")
            obs_long = gr.load(obs_long,use="G")
            nav_short = gr.load(nav_short)
            nav_medium = gr.load(nav_medium)
            nav_long = gr.load(nav_long)

            sv_list_gps_short = obs_short.sv.values
            sv_list_gps_medium = obs_medium.sv.values
            sv_list_gps_long = obs_long.sv.values
            sv_list = [sv for sv in sv_list_gps_long if sv in sv_list_gps_medium and sv in sv_list_gps_short]
            system = 'gps'
        
        print(sv_list)
        plot_spectrogram_distance_correlation(sv_list, date, material, obs_short, obs_medium, obs_long, system)
        
    elif len(files) == 4:
        obs_main, nav_main, obs_aux, nav_aux = files
        date = get_date(obs_main)
        material = get_material(obs_main)
        if 'galileo' in nav_main:
            obs_main = gr.load(obs_main,use="E")
            obs_aux = gr.load(obs_aux,use="E")
            nav_main = gr.load(nav_main)
            nav_aux = gr.load(nav_aux)
            sv_list_gal_main = obs_main.sv.values
            sv_list_gal_aux = obs_aux.sv.values
            sv_list = [sv for sv in sv_list_gal_main if sv in sv_list_gal_aux]
            system = 'galileo'
        else:
            obs_main = gr.load(obs_main,use="G")
            obs_aux = gr.load(obs_aux,use="G")
            nav_main = gr.load(nav_main)
            nav_aux = gr.load(nav_aux)

            sv_list_gps_main = obs_main.sv.values
            sv_list_gps_aux = obs_aux.sv.values
            sv_list = [sv for sv in sv_list_gps_main if sv in sv_list_gps_aux]
            system = 'gps'
        
        print(sv_list)
        plot_spectrogram_RHPvLHP(sv_list, date, material, obs_main, obs_aux, system)
        
if __name__ == '__main__':
    PERFORM_MOVING_AVERAGE = True

    PERFORM_WHOLE_AMPLITUDE_NORMALIZATION = False
    PERFORM_HALF_AMPLITUDE_NORMALIZATION = False
    PERFORM_GENERAL_AMPLITUDE_NORMALIZATION = False

    WINDOW_SIZE = 14

    if not WINDOW_SIZE % 2 == 0:
        WINDOW_SIZE += 1

    CN0_L1_SELECTOR = 'S1C'
    CN0_L2_SELECTOR = 'S2L'
    CN0_L5_SELECTOR = 'S5Q'
    CN0_E1_SELECTOR = "S1C"
    CN0_E5A_SELECTOR = "S5Q"
    CN0_E5B_SELECTOR = "S7Q"
    CN0_E5AB_SELECTOR = "S8Q"
    main()