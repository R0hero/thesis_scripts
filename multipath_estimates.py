import numpy as np
import georinex as gr
import matplotlib.pyplot as plt
import os
import re

# own functions
from sv_functions import calculate_multipath_error
from data_selector import data_selector

def get_material(file):
    if 'glass' in file:
        wall_type = 'glass'
    elif 'metal' in file:
        wall_type = 'metal'
    elif 'brick' in file:
        wall_type = 'brick'
    else:
        wall_type = 'clear'
    return wall_type

def get_date(file):
    pattern_date = r'\d{6}'
    match_date = re.search(pattern_date, file)[0]
    
    return match_date

def plot_subplot_for_comparable_plot(ax,time,mp,label_target,label_reference,set_ylabel=False,set_xlabel=False):
    ax.plot(time,mp,label=f'{label_target} with {label_reference} reference')
    
    ax.legend()
    if set_ylabel:
        ax.set_ylabel(f'Multipath error [m]')
    if set_xlabel:
        ax.set_xlabel('Time')
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))

def plot_comparable_plots(obs, sv, material, date, galileo_system=False):

    if galileo_system:
        pseudorange_E1 = obs.sel(sv=sv)[PSEUDORANGE_E1_SELECTOR]
        pseudorange_E5a = obs.sel(sv=sv)[PSEUDORANGE_E5A_SELECTOR]
        pseudorange_E5b = obs.sel(sv=sv)[PSEUDORANGE_E5B_SELECTOR]
        pseudorange_E5ab = obs.sel(sv=sv)[PSEUDORANGE_E5AB_SELECTOR]

        carrier_phase_E1 = obs.sel(sv=sv)[CARRIER_PHASE_E1_SELECTOR]
        carrier_phase_E5a = obs.sel(sv=sv)[CARRIER_PHASE_E5A_SELECTOR]
        carrier_phase_E5b = obs.sel(sv=sv)[CARRIER_PHASE_E5B_SELECTOR]
        carrier_phase_E5ab = obs.sel(sv=sv)[CARRIER_PHASE_E5AB_SELECTOR]

        mp_E1E5a = calculate_multipath_error(pseudorange_E1, FREQ_E1, carrier_phase_E1, FREQ_E5A, carrier_phase_E5a)
        np_E1E5b = calculate_multipath_error(pseudorange_E1, FREQ_E1, carrier_phase_E1, FREQ_E5B, carrier_phase_E5b)
        np_E1E5ab = calculate_multipath_error(pseudorange_E1, FREQ_E1, carrier_phase_E1, FREQ_E5AB, carrier_phase_E5ab)

        mp_E5aE1 = calculate_multipath_error(pseudorange_E5a, FREQ_E5A, carrier_phase_E5a, FREQ_E1, carrier_phase_E1)
        mp_E5aE5b = calculate_multipath_error(pseudorange_E5a, FREQ_E5A, carrier_phase_E5a, FREQ_E5B, carrier_phase_E5b)
        mp_E5aE5ab = calculate_multipath_error(pseudorange_E5a, FREQ_E5A, carrier_phase_E5a, FREQ_E5AB, carrier_phase_E5ab)

        mp_E5bE1 = calculate_multipath_error(pseudorange_E5b, FREQ_E5B, carrier_phase_E5b, FREQ_E1, carrier_phase_E1)
        mp_E5bE5a = calculate_multipath_error(pseudorange_E5b, FREQ_E5B, carrier_phase_E5b, FREQ_E5A, carrier_phase_E5a)
        mp_E5bE5ab = calculate_multipath_error(pseudorange_E5b, FREQ_E5B, carrier_phase_E5b, FREQ_E5AB, carrier_phase_E5ab)

        mp_E5abE1 = calculate_multipath_error(pseudorange_E5ab, FREQ_E5AB, carrier_phase_E5ab, FREQ_E1, carrier_phase_E1)
        mp_E5abE5a = calculate_multipath_error(pseudorange_E5ab, FREQ_E5AB, carrier_phase_E5ab, FREQ_E5A, carrier_phase_E5a)
        mp_E5abE5b = calculate_multipath_error(pseudorange_E5ab, FREQ_E5AB, carrier_phase_E5ab, FREQ_E5B, carrier_phase_E5b)

        fig, ax = plt.subplots(2,2,figsize=(8,6),sharex=True,sharey=True)
        fig.suptitle(f'SV {sv}')
        plot_subplot_for_comparable_plot(ax[0,0],obs.time.values,mp_E1E5a,"E1","E5a")
        plot_subplot_for_comparable_plot(ax[0,0],obs.time.values,np_E1E5b,"E1","E5b")
        plot_subplot_for_comparable_plot(ax[0,0],obs.time.values,np_E1E5ab,"E1","E5ab", set_ylabel=True)
        ax[0,0].axhline(y=0, linestyle='--', linewidth=0.8, color='black', alpha=0.5)

        plot_subplot_for_comparable_plot(ax[1,0],obs.time.values,mp_E5aE1,"E5a","E1")
        plot_subplot_for_comparable_plot(ax[1,0],obs.time.values,mp_E5aE5b,"E5a","E5b")
        plot_subplot_for_comparable_plot(ax[1,0],obs.time.values,mp_E5aE5ab,"E5a","E5ab", set_xlabel=True, set_ylabel=True)
        ax[1,0].axhline(y=0, linestyle='--', linewidth=0.8, color='black', alpha=0.5)
        
        plot_subplot_for_comparable_plot(ax[0,1],obs.time.values,mp_E5bE1,"E5b","E1")
        plot_subplot_for_comparable_plot(ax[0,1],obs.time.values,mp_E5bE5a,"E5b","E5a")
        plot_subplot_for_comparable_plot(ax[0,1],obs.time.values,mp_E5bE5ab,"E5b","E5ab")
        ax[0,1].axhline(y=0, linestyle='--', linewidth=0.8, color='black', alpha=0.5)
        
        plot_subplot_for_comparable_plot(ax[1,1],obs.time.values,mp_E5abE1,"E5ab","E1")
        plot_subplot_for_comparable_plot(ax[1,1],obs.time.values,mp_E5abE5a,"E5ab","E5a")
        plot_subplot_for_comparable_plot(ax[1,1],obs.time.values,mp_E5abE5b,"E5ab","E5b", set_xlabel=True)
        ax[1,1].axhline(y=0, linestyle='--', linewidth=0.8, color='black', alpha=0.5)
    else:
        pseudorange_L1 = obs[PSEUDORANGE_L1_SELECTOR].sel(sv=sv).values
        pseudorange_L2 = obs[PSEUDORANGE_L2_SELECTOR].sel(sv=sv).values
        pseudorange_L5 = obs[PSEUDORANGE_L5_SELECTOR].sel(sv=sv).values

        carrier_phase_L1 = obs[CARRIER_PHASE_L1_SELECTOR].sel(sv=sv).values # in cycles
        carrier_phase_L2 = obs[CARRIER_PHASE_L2_SELECTOR].sel(sv=sv).values # in cycles
        carrier_phase_L5 = obs[CARRIER_PHASE_L5_SELECTOR].sel(sv=sv).values # in cycles

        mp_L1L2 = calculate_multipath_error(pseudorange_L1, FREQ_L1, carrier_phase_L1, FREQ_L2, carrier_phase_L2)
        mp_L1L5 = calculate_multipath_error(pseudorange_L1, FREQ_L1, carrier_phase_L1, FREQ_L5, carrier_phase_L5)
        mp_L2L1 = calculate_multipath_error(pseudorange_L2, FREQ_L2, carrier_phase_L2, FREQ_L1, carrier_phase_L1)
        mp_L2L5 = calculate_multipath_error(pseudorange_L2, FREQ_L2, carrier_phase_L2, FREQ_L5, carrier_phase_L5)
        mp_L5L1 = calculate_multipath_error(pseudorange_L5, FREQ_L5, carrier_phase_L5, FREQ_L1, carrier_phase_L1)
        mp_L5L2 = calculate_multipath_error(pseudorange_L5, FREQ_L5, carrier_phase_L5, FREQ_L2, carrier_phase_L2)

        fig, ax = plt.subplots(3,1,figsize=(8,8),sharex=True,sharey=True)
        fig.suptitle(f'SV {sv}')
        plot_subplot_for_comparable_plot(ax[0],obs.time.values,mp_L1L2,"L1","L2")
        plot_subplot_for_comparable_plot(ax[0],obs.time.values,mp_L1L5,"L1","L5", set_ylabel=True)
        ax[0].axhline(y=0, linestyle='--', linewidth=0.8, color='black', alpha=0.5)
        plot_subplot_for_comparable_plot(ax[1],obs.time.values,mp_L2L1,"L2","L1")
        plot_subplot_for_comparable_plot(ax[1],obs.time.values,mp_L2L5,"L2","L5", set_ylabel=True)
        ax[1].axhline(y=0, linestyle='--', linewidth=0.8, color='black', alpha=0.5)
        plot_subplot_for_comparable_plot(ax[2],obs.time.values,mp_L5L1,"L5","L1")
        plot_subplot_for_comparable_plot(ax[2],obs.time.values,mp_L5L2,"L5","L2", set_xlabel=True, set_ylabel=True)
        ax[2].axhline(y=0, linestyle='--', linewidth=0.8, color='black', alpha=0.5)
    
    plt.tight_layout()

    if SAVE_COMPARABLE_PLOTS:
        plt.savefig(f'..{os.sep}figs{os.sep}multipath_errors{os.sep}multipath_errors_comparing_frequencies_{sv}_{material}_{date}.png')
        plt.savefig(f'..{os.sep}figs{os.sep}multipath_errors{os.sep}multipath_errors_comparing_frequencies_{sv}_{material}_{date}.pdf')
    if SHOW_COMPARABLE_PLOTS:
        plt.show()


def plot_singular_plots(obs, sv, material, date, galileo_system=False):

    plt.figure(figsize=(8,4))
    plt.title(f'SV {sv}')
    if galileo_system:
        pseudorange_E1 = obs.sel(sv=sv)[PSEUDORANGE_E1_SELECTOR].values
        pseudorange_E5a = obs.sel(sv=sv)[PSEUDORANGE_E5A_SELECTOR].values
        pseudorange_E5b = obs.sel(sv=sv)[PSEUDORANGE_E5B_SELECTOR].values
        pseudorange_E5ab = obs.sel(sv=sv)[PSEUDORANGE_E5AB_SELECTOR].values

        carrier_phase_E1 = obs.sel(sv=sv)[CARRIER_PHASE_E1_SELECTOR].values
        carrier_phase_E5a = obs.sel(sv=sv)[CARRIER_PHASE_E5A_SELECTOR].values
        carrier_phase_E5b = obs.sel(sv=sv)[CARRIER_PHASE_E5B_SELECTOR].values
        carrier_phase_E5ab = obs.sel(sv=sv)[CARRIER_PHASE_E5AB_SELECTOR].values

        mp_E1E5a = calculate_multipath_error(pseudorange_E1, FREQ_E1, carrier_phase_E1, FREQ_E5A, carrier_phase_E5a)
        mp_E5aE1 = calculate_multipath_error(pseudorange_E5a, FREQ_E5A, carrier_phase_E5a, FREQ_E1, carrier_phase_E1)
        mp_E5bE1 = calculate_multipath_error(pseudorange_E5b, FREQ_E5B, carrier_phase_E5b, FREQ_E1, carrier_phase_E1)
        mp_E5abE1 = calculate_multipath_error(pseudorange_E5ab, FREQ_E5AB, carrier_phase_E5ab, FREQ_E1, carrier_phase_E1)

        # print(sv)
        # print('mp_E1E5a')
        # print(max(np.nanmax(mp_E1E5a),np.abs(np.nanmin(mp_E1E5a))))
        # print('mp_E5aE1')
        # print(max(np.nanmax(mp_E5aE1),np.abs(np.nanmin(mp_E5aE1))))
        # print('mp_E5bE1')
        # print(max(np.nanmax(mp_E5bE1),np.abs(np.nanmin(mp_E5bE1))))
        # print('mp_E5abE1')
        # print(max(np.nanmax(mp_E5abE1),np.abs(np.nanmin(mp_E5abE1))))

        plt.plot(obs.time.values, mp_E1E5a, label=f'E1 with E5a reference, std: {np.nanstd(mp_E1E5a):.2f} m')
        plt.plot(obs.time.values, mp_E5aE1, label=f'E5a with E1 reference, std: {np.nanstd(mp_E5aE1):.2f} m')
        plt.plot(obs.time.values, mp_E5bE1, label=f'E5b with E1 reference, std: {np.nanstd(mp_E5bE1):.2f} m')
        plt.plot(obs.time.values, mp_E5abE1, label=f'E5ab with E1 reference, std: {np.nanstd(mp_E5abE1):.2f} m')
    else:
        pseudorange_L1 = obs[PSEUDORANGE_L1_SELECTOR].sel(sv=sv).values
        pseudorange_L2 = obs[PSEUDORANGE_L2_SELECTOR].sel(sv=sv).values
        pseudorange_L5 = obs[PSEUDORANGE_L5_SELECTOR].sel(sv=sv).values

        carrier_phase_L1 = obs[CARRIER_PHASE_L1_SELECTOR].sel(sv=sv).values # in cycles
        carrier_phase_L2 = obs[CARRIER_PHASE_L2_SELECTOR].sel(sv=sv).values # in cycles
        carrier_phase_L5 = obs[CARRIER_PHASE_L5_SELECTOR].sel(sv=sv).values # in cycles

        mp_L1L2 = calculate_multipath_error(pseudorange_L1, FREQ_L1, carrier_phase_L1, FREQ_L2, carrier_phase_L2)
        mp_L2L1 = calculate_multipath_error(pseudorange_L2, FREQ_L2, carrier_phase_L2, FREQ_L1, carrier_phase_L1)
        mp_L5L1 = calculate_multipath_error(pseudorange_L5, FREQ_L5, carrier_phase_L5, FREQ_L1, carrier_phase_L1)

        # print(sv)
        # print('mp_L1L2')
        # print(max(np.nanmax(mp_L1L2),np.abs(np.nanmin(mp_L1L2))))
        # print('mp_L2L1')
        # print(max(np.nanmax(mp_L2L1),np.abs(np.nanmin(mp_L2L1))))
        # print('mp_L5L1')
        # print(max(np.nanmax(mp_L5L1),np.abs(np.nanmin(mp_L5L1))))

        plt.plot(obs.time.values,mp_L1L2,label=f'L1 with L2 reference, std: {np.nanstd(mp_L1L2):.2f} m')
        plt.plot(obs.time.values,mp_L2L1,label=f'L2 with L1 reference, std: {np.nanstd(mp_L2L1):.2f} m')
        plt.plot(obs.time.values,mp_L5L1,label=f'L5 with L1 reference, std: {np.nanstd(mp_L5L1):.2f} m')

    plt.axhline(y=0, linestyle='--', linewidth=0.8, color='black', alpha=0.5)
    plt.ylabel('Multipath error [m]')
    plt.xlabel('Time')
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
    plt.legend()
    
    if SAVE_SINGULAR_PLOTS:
        plt.savefig(f'..{os.sep}figs{os.sep}multipath_errors{os.sep}multipath_errors_singular_plot_{sv}_{material}_{date}.png')
        plt.savefig(f'..{os.sep}figs{os.sep}multipath_errors{os.sep}multipath_errors_singular_plot_{sv}_{material}_{date}.pdf')
    if SHOW_SINGULAR_PLOTS:
        plt.show()

def compare_distances(obs_short, obs_medium, obs_long, sv, material, date, galileo_system=False):
    if galileo_system:
        pseudorange_E1 = obs_short.sel(sv=sv)[PSEUDORANGE_E1_SELECTOR]
        pseudorange_E5a = obs_short.sel(sv=sv)[PSEUDORANGE_E5A_SELECTOR]
        pseudorange_E5b = obs_short.sel(sv=sv)[PSEUDORANGE_E5B_SELECTOR]
        pseudorange_E5ab = obs_short.sel(sv=sv)[PSEUDORANGE_E5AB_SELECTOR]

        carrier_phase_E1 = obs_short.sel(sv=sv)[CARRIER_PHASE_E1_SELECTOR]
        carrier_phase_E5a = obs_short.sel(sv=sv)[CARRIER_PHASE_E5A_SELECTOR]
        carrier_phase_E5b = obs_short.sel(sv=sv)[CARRIER_PHASE_E5B_SELECTOR]
        carrier_phase_E5ab = obs_short.sel(sv=sv)[CARRIER_PHASE_E5AB_SELECTOR]

        mp_E1E5a_short = calculate_multipath_error(pseudorange_E1, FREQ_E1, carrier_phase_E1, FREQ_E5A, carrier_phase_E5a)
        mp_E5aE1_short = calculate_multipath_error(pseudorange_E5a, FREQ_E5A, carrier_phase_E5a, FREQ_E1, carrier_phase_E1)
        mp_E5bE1_short = calculate_multipath_error(pseudorange_E5b, FREQ_E5B, carrier_phase_E5b, FREQ_E1, carrier_phase_E1)
        mp_E5abE1_short = calculate_multipath_error(pseudorange_E5ab, FREQ_E5AB, carrier_phase_E5ab, FREQ_E1, carrier_phase_E1)

        pseudorange_E1 = obs_medium.sel(sv=sv)[PSEUDORANGE_E1_SELECTOR]
        pseudorange_E5a = obs_medium.sel(sv=sv)[PSEUDORANGE_E5A_SELECTOR]
        pseudorange_E5b = obs_medium.sel(sv=sv)[PSEUDORANGE_E5B_SELECTOR]
        pseudorange_E5ab = obs_medium.sel(sv=sv)[PSEUDORANGE_E5AB_SELECTOR]

        carrier_phase_E1 = obs_medium.sel(sv=sv)[CARRIER_PHASE_E1_SELECTOR]
        carrier_phase_E5a = obs_medium.sel(sv=sv)[CARRIER_PHASE_E5A_SELECTOR]
        carrier_phase_E5b = obs_medium.sel(sv=sv)[CARRIER_PHASE_E5B_SELECTOR]
        carrier_phase_E5ab = obs_medium.sel(sv=sv)[CARRIER_PHASE_E5AB_SELECTOR]

        mp_E1E5a_medium = calculate_multipath_error(pseudorange_E1, FREQ_E1, carrier_phase_E1, FREQ_E5A, carrier_phase_E5a)
        mp_E5aE1_medium = calculate_multipath_error(pseudorange_E5a, FREQ_E5A, carrier_phase_E5a, FREQ_E1, carrier_phase_E1)
        mp_E5bE1_medium = calculate_multipath_error(pseudorange_E5b, FREQ_E5B, carrier_phase_E5b, FREQ_E1, carrier_phase_E1)
        mp_E5abE1_medium = calculate_multipath_error(pseudorange_E5ab, FREQ_E5AB, carrier_phase_E5ab, FREQ_E1, carrier_phase_E1)

        pseudorange_E1 = obs_long.sel(sv=sv)[PSEUDORANGE_E1_SELECTOR]
        pseudorange_E5a = obs_long.sel(sv=sv)[PSEUDORANGE_E5A_SELECTOR]
        pseudorange_E5b = obs_long.sel(sv=sv)[PSEUDORANGE_E5B_SELECTOR]
        pseudorange_E5ab = obs_long.sel(sv=sv)[PSEUDORANGE_E5AB_SELECTOR]

        carrier_phase_E1 = obs_long.sel(sv=sv)[CARRIER_PHASE_E1_SELECTOR]
        carrier_phase_E5a = obs_long.sel(sv=sv)[CARRIER_PHASE_E5A_SELECTOR]
        carrier_phase_E5b = obs_long.sel(sv=sv)[CARRIER_PHASE_E5B_SELECTOR]
        carrier_phase_E5ab = obs_long.sel(sv=sv)[CARRIER_PHASE_E5AB_SELECTOR]

        mp_E1E5a_long = calculate_multipath_error(pseudorange_E1, FREQ_E1, carrier_phase_E1, FREQ_E5A, carrier_phase_E5a)
        mp_E5aE1_long = calculate_multipath_error(pseudorange_E5a, FREQ_E5A, carrier_phase_E5a, FREQ_E1, carrier_phase_E1)
        mp_E5bE1_long = calculate_multipath_error(pseudorange_E5b, FREQ_E5B, carrier_phase_E5b, FREQ_E1, carrier_phase_E1)
        mp_E5abE1_long = calculate_multipath_error(pseudorange_E5ab, FREQ_E5AB, carrier_phase_E5ab, FREQ_E1, carrier_phase_E1)

        fig,ax = plt.subplots(3,1,figsize=(8,9), sharex=True, sharey=True)
        fig.suptitle(f'SV {sv}')
        ax[0].plot(obs_short.time.values,mp_E1E5a_short,label=f'E1 with E5a reference, std: {np.nanstd(mp_E1E5a_short):.2f} m')
        ax[0].plot(obs_short.time.values,mp_E5aE1_short,label=f'E5a with E1 reference, std: {np.nanstd(mp_E5aE1_short):.2f} m')
        ax[0].plot(obs_short.time.values,mp_E5bE1_short,label=f'E5b with E1 reference, std: {np.nanstd(mp_E5bE1_short):.2f} m')
        ax[0].plot(obs_short.time.values,mp_E5abE1_short,label=f'E5ab with E1 reference, std: {np.nanstd(mp_E5abE1_short):.2f} m')
        ax[0].axhline(y=0, linestyle='--', linewidth=0.8, color='black', alpha=0.5)
        ax[0].set_ylabel('Multipath error [m]')
        ax[0].set_title('2,5m from wall')
        ax[0].legend()

        ax[1].plot(obs_medium.time.values,mp_E1E5a_medium,label=f'E1 with E5a reference, std: {np.nanstd(mp_E1E5a_medium):.2f} m')
        ax[1].plot(obs_medium.time.values,mp_E5aE1_medium,label=f'E5a with E1 reference, std: {np.nanstd(mp_E5aE1_medium):.2f} m')
        ax[1].plot(obs_medium.time.values,mp_E5bE1_medium,label=f'E5b with E1 reference, std: {np.nanstd(mp_E5bE1_medium):.2f} m')
        ax[1].plot(obs_medium.time.values,mp_E5abE1_medium,label=f'E5ab with E1 reference, std: {np.nanstd(mp_E5abE1_medium):.2f} m')
        ax[1].axhline(y=0, linestyle='--', linewidth=0.8, color='black', alpha=0.5)
        ax[1].set_ylabel('Multipath error [m]')
        ax[1].set_title('5m from wall')
        ax[1].legend()

        ax[2].plot(obs_long.time.values,mp_E1E5a_long,label=f'E1 with E5a reference, std: {np.nanstd(mp_E1E5a_long):.2f} m')
        ax[2].plot(obs_long.time.values,mp_E5aE1_long,label=f'E5a with E1 reference, std: {np.nanstd(mp_E5aE1_long):.2f} m')
        ax[2].plot(obs_long.time.values,mp_E5bE1_long,label=f'E5b with E1 reference, std: {np.nanstd(mp_E5bE1_long):.2f} m')
        ax[2].plot(obs_long.time.values,mp_E5abE1_long,label=f'E5ab with E1 reference, std: {np.nanstd(mp_E5abE1_long):.2f} m')
        ax[2].axhline(y=0, linestyle='--', linewidth=0.8, color='black', alpha=0.5)
        ax[2].set_ylabel('Multipath error [m]')
        ax[2].set_title('10m from wall')
        ax[2].set_xlabel('Time')
        ax[2].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        ax[2].legend()

        plt.tight_layout()
    else:
        pseudorange_L1 = obs_short.sel(sv=sv)[PSEUDORANGE_L1_SELECTOR]
        pseudorange_L2 = obs_short.sel(sv=sv)[PSEUDORANGE_L2_SELECTOR]
        pseudorange_L5 = obs_short.sel(sv=sv)[PSEUDORANGE_L5_SELECTOR]

        carrier_phase_L1 = obs_short.sel(sv=sv)[CARRIER_PHASE_L1_SELECTOR]
        carrier_phase_L2 = obs_short.sel(sv=sv)[CARRIER_PHASE_L2_SELECTOR]
        carrier_phase_L5 = obs_short.sel(sv=sv)[CARRIER_PHASE_L5_SELECTOR]

        mp_L1L2_short = calculate_multipath_error(pseudorange_L1, FREQ_L1, carrier_phase_L1, FREQ_L2, carrier_phase_L2)
        mp_L2L1_short = calculate_multipath_error(pseudorange_L2, FREQ_L2, carrier_phase_L2, FREQ_L1, carrier_phase_L1)
        mp_L5L1_short = calculate_multipath_error(pseudorange_L5, FREQ_L5, carrier_phase_L5, FREQ_L1, carrier_phase_L1)

        pseudorange_L1 = obs_medium.sel(sv=sv)[PSEUDORANGE_L1_SELECTOR]
        pseudorange_L2 = obs_medium.sel(sv=sv)[PSEUDORANGE_L2_SELECTOR]
        pseudorange_L5 = obs_medium.sel(sv=sv)[PSEUDORANGE_L5_SELECTOR]

        carrier_phase_L1 = obs_medium.sel(sv=sv)[CARRIER_PHASE_L1_SELECTOR]
        carrier_phase_L2 = obs_medium.sel(sv=sv)[CARRIER_PHASE_L2_SELECTOR]
        carrier_phase_L5 = obs_medium.sel(sv=sv)[CARRIER_PHASE_L5_SELECTOR]

        mp_L1L2_medium = calculate_multipath_error(pseudorange_L1, FREQ_L1, carrier_phase_L1, FREQ_L2, carrier_phase_L2)
        mp_L2L1_medium = calculate_multipath_error(pseudorange_L2, FREQ_L2, carrier_phase_L2, FREQ_L1, carrier_phase_L1)
        mp_L5L1_medium = calculate_multipath_error(pseudorange_L5, FREQ_L5, carrier_phase_L5, FREQ_L1, carrier_phase_L1)

        pseudorange_L1 = obs_long.sel(sv=sv)[PSEUDORANGE_L1_SELECTOR]
        pseudorange_L2 = obs_long.sel(sv=sv)[PSEUDORANGE_L2_SELECTOR]
        pseudorange_L5 = obs_long.sel(sv=sv)[PSEUDORANGE_L5_SELECTOR]

        carrier_phase_L1 = obs_long.sel(sv=sv)[CARRIER_PHASE_L1_SELECTOR]
        carrier_phase_L2 = obs_long.sel(sv=sv)[CARRIER_PHASE_L2_SELECTOR]
        carrier_phase_L5 = obs_long.sel(sv=sv)[CARRIER_PHASE_L5_SELECTOR]

        mp_L1L2_long = calculate_multipath_error(pseudorange_L1, FREQ_L1, carrier_phase_L1, FREQ_L2, carrier_phase_L2)
        mp_L2L1_long = calculate_multipath_error(pseudorange_L2, FREQ_L2, carrier_phase_L2, FREQ_L1, carrier_phase_L1)
        mp_L5L1_long = calculate_multipath_error(pseudorange_L5, FREQ_L5, carrier_phase_L5, FREQ_L1, carrier_phase_L1)

        fig,ax = plt.subplots(3,1,figsize=(8,9), sharex=True, sharey=True)
        fig.suptitle(f'SV {sv}')
        ax[0].plot(obs_short.time.values,mp_L1L2_short,label=f'L1 with L2 reference, std: {np.nanstd(mp_L1L2_short):.2f} m')
        ax[0].plot(obs_short.time.values,mp_L2L1_short,label=f'L2 with L1 reference, std: {np.nanstd(mp_L2L1_short):.2f} m')
        ax[0].plot(obs_short.time.values,mp_L5L1_short,label=f'L5 with L1 reference, std: {np.nanstd(mp_L5L1_short):.2f} m')
        ax[0].axhline(y=0, linestyle='--', linewidth=0.8, color='black', alpha=0.5)
        ax[0].set_ylabel('Multipath error [m]')
        ax[0].set_title('2,5m from wall')
        ax[0].legend()

        ax[1].plot(obs_medium.time.values,mp_L1L2_medium,label=f'L1 with L2 reference, std: {np.nanstd(mp_L1L2_medium):.2f} m')
        ax[1].plot(obs_medium.time.values,mp_L2L1_medium,label=f'L2 with L1 reference, std: {np.nanstd(mp_L2L1_medium):.2f} m')
        ax[1].plot(obs_medium.time.values,mp_L5L1_medium,label=f'L5 with L1 reference, std: {np.nanstd(mp_L5L1_medium):.2f} m')
        ax[1].axhline(y=0, linestyle='--', linewidth=0.8, color='black', alpha=0.5)
        ax[1].set_ylabel('Multipath error [m]')
        ax[1].set_title('5m from wall')
        ax[1].legend()

        ax[2].plot(obs_long.time.values,mp_L1L2_long,label=f'L1 with L2 reference, std: {np.nanstd(mp_L1L2_long):.2f} m')
        ax[2].plot(obs_long.time.values,mp_L2L1_long,label=f'L2 with L1 reference, std: {np.nanstd(mp_L2L1_long):.2f} m')
        ax[2].plot(obs_long.time.values,mp_L5L1_long,label=f'L5 with L1 reference, std: {np.nanstd(mp_L5L1_long):.2f} m')
        ax[2].axhline(y=0, linestyle='--', linewidth=0.8, color='black', alpha=0.5)
        ax[2].set_ylabel('Multipath error [m]')
        ax[2].set_title('10m from wall')
        ax[2].set_xlabel('Time')
        ax[2].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        ax[2].legend()

        plt.tight_layout()
    
    if SAVE_DISTANCE_PLOTS:
        plt.savefig(f'..{os.sep}figs{os.sep}multipath_errors{os.sep}multipath_errors_distance_correlation_{sv}_{material}_{date}.png')
        plt.savefig(f'..{os.sep}figs{os.sep}multipath_errors{os.sep}multipath_errors_distance_correlation_{sv}_{material}_{date}.pdf')
    if SHOW_DISTANCE_PLOTS:
        plt.show()

def main():
    files = data_selector()

    if 'galileo' in files[1]:
        galileo_system = True
    else:
        galileo_system = False

    if len(files) == 4:
        obs, _, _, _ = files
        material = get_material(obs)
        date = get_date(obs)
        if galileo_system:
            obs = gr.load(obs,use="E")
        else:
            obs = gr.load(obs,use="G")
    elif len(files) == 6:
        obs_short, _, obs_medium, _, obs_long, _ = files
        material = get_material(obs_short)
        date = get_date(obs_short)
        if galileo_system:
            obs_short = gr.load(obs_short,use="E")
            obs_medium = gr.load(obs_medium,use="E")
            obs_long = gr.load(obs_long,use="E")
        else:
            obs_short = gr.load(obs_short,use="G")
            obs_medium = gr.load(obs_medium,use="G")
            obs_long = gr.load(obs_long,use="G")
    
    if len(files) == 4:
        sv_list = obs.sv.values
        for _, sv in enumerate(sv_list):
            plot_comparable_plots(obs, sv, material, date, galileo_system)
            plot_singular_plots(obs, sv, material, date, galileo_system)
    elif len(files) == 6:
        sv_list_short = obs_short.sv.values
        sv_list_medium = obs_medium.sv.values
        sv_list_long = obs_long.sv.values
        sv_list = [sv for sv in sv_list_short if sv in sv_list_medium and sv in sv_list_long]
        for _, sv in enumerate(sv_list):
            compare_distances(obs_short, obs_medium, obs_long, sv, material, date, galileo_system)


if __name__ == '__main__':
    # GPS
    PSEUDORANGE_L1_SELECTOR = 'C1C'
    PSEUDORANGE_L2_SELECTOR = 'C2L'
    PSEUDORANGE_L5_SELECTOR = 'C5Q'
    # Galileo
    PSEUDORANGE_E1_SELECTOR = "C1C"
    PSEUDORANGE_E5A_SELECTOR = "C5Q"
    PSEUDORANGE_E5B_SELECTOR = "C7Q"
    PSEUDORANGE_E5AB_SELECTOR = "C8Q"
    # GPS
    CARRIER_PHASE_L1_SELECTOR = 'L1C'
    CARRIER_PHASE_L2_SELECTOR = 'L2L'
    CARRIER_PHASE_L5_SELECTOR = 'L5Q'
    # Galileo
    CARRIER_PHASE_E1_SELECTOR = "L1C"
    CARRIER_PHASE_E5A_SELECTOR = "L5Q"
    CARRIER_PHASE_E5B_SELECTOR = "L7Q"
    CARRIER_PHASE_E5AB_SELECTOR = "L8Q"

    FREQ_L1 = 10.23e06 * 154 # Hz
    FREQ_L2 = 10.23e06 * 120 # Hz
    FREQ_L5 = 10.23e06 * 115 # Hz

    FREQ_E1 = 10.23e06 * 154 # Hz
    FREQ_E5A = 10.23e06 * 115 # Hz
    FREQ_E5B = 10.23e06 * 118 # Hz
    FREQ_E5AB = (FREQ_E5A + FREQ_E5B)/2 # Hz
    FREQ_E6 = 10.23e06 * 125 # Hz

    SAVE_COMPARABLE_PLOTS = True
    SHOW_COMPARABLE_PLOTS = False
    SAVE_SINGULAR_PLOTS = True
    SHOW_SINGULAR_PLOTS = False
    SAVE_DISTANCE_PLOTS = True
    SHOW_DISTANCE_PLOTS = False

    main()