import georinex as gr
import numpy as np
import matplotlib.pyplot as plt

from sv_functions import ecef_to_lla, calcSatPos, R1, R3

def get_common_svlist(main,aux):

    sv_list_main = main.sv.values
    sv_list_aux = aux.sv.values

    idx = np.ones_like(sv_list_main,dtype=bool)

    for i in range(len(sv_list_main)):
        if sv_list_main[i] not in sv_list_aux:
            idx[i] = False
    return sv_list_main[idx]

def plot_ratios(sv_list,material):

    if material == 'brick':
        obs_main = OBS_MAIN_BRICK
        obs_aux = OBS_AUX_BRICK
    elif material == 'glass':
        obs_main = OBS_MAIN_GLASS
        obs_aux = OBS_AUX_GLASS
    elif material == 'metal':
        obs_main = OBS_MAIN_METAL
        obs_aux = OBS_AUX_METAL
    else:
        obs_main = OBS_MAIN_CLEAR
        obs_aux = OBS_AUX_CLEAR


    for sv in sv_list:
        CN0_L1_main = obs_main.sel(sv=sv)['S1C'].values
        CN0_L2_main = obs_main.sel(sv=sv)['S2L'].values
        CN0_L5_main = obs_main.sel(sv=sv)['S5Q'].values

        CN0_L1_aux = obs_aux.sel(sv=sv)['S1C'].values
        CN0_L2_aux = obs_aux.sel(sv=sv)['S2L'].values
        CN0_L5_aux = obs_aux.sel(sv=sv)['S5Q'].values

        if PLOT_RAW_CN0:
            fig, ax = plt.subplots(3,1,figsize=(8,8),sharex=True)
            if not material == '':
                fig.suptitle(f'SV {sv}\n{material} 10m')
            else:
                fig.suptitle(f'SV {sv}')
            ax[0].plot(obs_main.sel(sv=sv).time.values, CN0_L1_main, color='blue', label=f'C/N_0 L1 RHP')
            ax[0].plot(obs_main.sel(sv=sv).time.values, CN0_L1_aux, color='red', label=f'C/N_0 L1 LHP')
            ax[0].set_ylabel(r'$C/N_0$ [dB-Hz]')
            ax[0].legend()

            ax[1].plot(obs_main.sel(sv=sv).time.values, CN0_L2_main, color='blue', label=f'C/N_0 L2 RHP')
            ax[1].plot(obs_main.sel(sv=sv).time.values, CN0_L2_aux, color='red', label=f'C/N_0 L2 LHP')
            ax[1].set_ylabel(r'$C/N_0$ [dB-Hz]')
            ax[1].legend()

            ax[2].plot(obs_main.sel(sv=sv).time.values, CN0_L5_main, color='blue', label=f'C/N_0 L5 RHP')
            ax[2].plot(obs_main.sel(sv=sv).time.values, CN0_L5_aux, color='red', label=f'C/N_0 L5 LHP')
            ax[2].set_ylabel(r'$C/N_0$ [dB-Hz]')
            ax[2].legend()
            plt.show()

        if PLOT_RATIO_SINGLE_SV:
            fig, ax = plt.subplots(3,1,figsize=(8,8),sharex=True,sharey=True)
            if not material == '':
                fig.suptitle(f'SV {sv}\n{material} 10m')
            else:
                fig.suptitle(f'SV {sv}')
            ax[0].plot(obs_main.sel(sv=sv).time.values, CN0_L1_main/CN0_L1_aux, color='green', label=f'C/N_0 L1 ratio RHP/LHP')
            ax[0].set_ylabel(r'$C/N_0$ Ratio')
            ax[0].legend()

            ax[1].plot(obs_main.sel(sv=sv).time.values, CN0_L2_main/CN0_L2_aux, color='green', label=f'C/N_0 L2 ratio RHP/LHP')
            ax[1].set_ylabel(r'$C/N_0$ Ratio')
            ax[1].legend()

            ax[2].plot(obs_main.sel(sv=sv).time.values, CN0_L5_main/CN0_L5_aux, color='green', label=f'C/N_0 L5 ratio RHP/LHP')
            ax[2].set_ylabel(r'$C/N_0$ Ratio')
            ax[2].legend()
            plt.show()

def get_el_az(material):

    if material == 'brick':
        obs = OBS_MAIN_BRICK
        nav = NAV_MAIN_BRICK
    elif material == 'glass':
        obs = OBS_MAIN_GLASS
        nav = NAV_MAIN_GLASS
    elif material == 'metal':
        obs = OBS_MAIN_METAL
        nav = NAV_MAIN_METAL
    else:
        obs = OBS_MAIN_CLEAR
        nav = NAV_MAIN_CLEAR
    sv_list = obs.sv.values

    skip_sv = []
    while True:
        sv_list = [sv for sv in sv_list if sv not in skip_sv]
        try:
            ecef_pos = np.array(obs.position)
            lla_pos = ecef_to_lla(ecef_pos[0],ecef_pos[1],ecef_pos[2])

            duration = (obs.time.values[-1] - obs.time.values[0]).astype('timedelta64[s]').astype(np.int64)

            t = np.linspace(0,duration,20) + (np.int64(np.nanmin(nav.TransTime.values)))

            # initialize 
            az = np.zeros((len(sv_list),len(t)))
            el = np.zeros_like(az)
            zen = np.zeros_like(az)

            R_L = R1(90-lla_pos[0],deg=True)@R3(lla_pos[1]+90,deg=True)
            ## computing
            for t_i in range(len(t)):
                # calculate satellite positions
                sat_positions, _, _ = calcSatPos(nav, t[t_i], sv_list)
                for j in range(np.shape(sat_positions)[1]):
                    # calculate ENU coordinates of satellite
                    sat_ENU = (R_L @ (sat_positions[:,j].reshape((-1,1)) - ecef_pos.reshape((-1,1))))
                    
                    # calculate azimuth and zenith
                    azimuth = np.arctan2(sat_ENU[0], sat_ENU[1])
                    azimuth = np.rad2deg(azimuth[0])
                    zenith = np.arccos(sat_ENU[2] / np.sqrt(sat_ENU[0]**2 + sat_ENU[1]**2 + sat_ENU[2]**2))  
                    zenith = np.rad2deg(zenith[0])
                    zen[j,t_i] = zenith

                    if azimuth < 0:
                        azimuth = 360 + azimuth
                    
                    if ((90-zenith) > 0):
                        az[j,t_i] = azimuth
                        el[j,t_i] = 90-zenith
                    else:
                        az[j,t_i] = np.nan
                        el[j,t_i] = np.nan
            break
        except KeyError:
            try:
                for sv in sv_list:
                    nav.sel(sv=sv)
            except KeyError:
                skip_sv.append(sv)
    return el, az

def find_time_discontinuities(datetime_array):
    # Calculate time differences
    time_diff = np.diff(datetime_array) / np.timedelta64(1, 's')  # Convert to seconds
    
    # Find indexes where time difference is greater than 1 second
    discontinuity_indexes = np.where(time_diff > 1)[0] + 1
    
    return discontinuity_indexes

def ensure_same_length(CN0_main, CN0_aux, time_values):
    while True:
        if len(CN0_main) - len(CN0_aux) > 1:
            CN0_aux = np.insert(CN0_aux, find_time_discontinuities(time_values), np.nan)
        elif len(CN0_main) - len(CN0_aux) == 1:
            CN0_aux = np.append(CN0_aux, np.nan)
        else:
            return CN0_aux

def get_ratio_single_sv(material,sv):
    
    if material == 'brick':
        obs_main = OBS_MAIN_BRICK
        obs_aux = OBS_AUX_BRICK
    elif material == 'glass':
        obs_main = OBS_MAIN_GLASS
        obs_aux = OBS_AUX_GLASS
    elif material == 'metal':
        obs_main = OBS_MAIN_METAL
        obs_aux = OBS_AUX_METAL
    else:
        obs_main = OBS_MAIN_CLEAR
        obs_aux = OBS_AUX_CLEAR
    
    CN0_L1_main = obs_main.sel(sv=sv)['S1C'].values
    CN0_L2_main = obs_main.sel(sv=sv)['S2L'].values
    CN0_L5_main = obs_main.sel(sv=sv)['S5Q'].values

    CN0_L1_aux = obs_aux.sel(sv=sv)['S1C'].values
    CN0_L2_aux = obs_aux.sel(sv=sv)['S2L'].values
    CN0_L5_aux = obs_aux.sel(sv=sv)['S5Q'].values
    
    if len(CN0_L1_aux) == len(CN0_L1_main):
        CN0_L1_ratio = CN0_L1_main / CN0_L1_aux
        CN0_L2_ratio = CN0_L2_main / CN0_L2_aux
        CN0_L5_ratio = CN0_L5_main / CN0_L5_aux
    else:
        if not (obs_main.time.values[0] - obs_aux.time.values[0]) == 0:
            try:
                start_idx = np.where(obs_main.time.values[0] == obs_aux.time.values)[0][0]
            except IndexError:
                start_idx = np.where(obs_aux.time.values[0] == obs_main.time.values)[0][0]
            if (obs_main.time.values[0] - obs_aux.time.values[0]) < 0:
                start_idx_main = start_idx 
                start_idx_aux = 0 
            elif (obs_main.time.values[0] - obs_aux.time.values[0]) > 0:
                start_idx_aux = start_idx 
                start_idx_main = 0
        if not (obs_main.time.values[-1] - obs_aux.time.values[-1]) == 0:
            try:
                end_idx = np.where(obs_main.time.values[-1] == obs_aux.time.values)[0][0]
            except IndexError:
                end_idx = np.where(obs_aux.time.values[-1] == obs_main.time.values)[0][0]
            if (obs_main.time.values[-1] - obs_aux.time.values[-1]) > 0:
                end_idx_main = end_idx 
                end_idx_aux = -1 
            elif (obs_main.time.values[-1] - obs_aux.time.values[-1]) < 0:
                end_idx_aux = end_idx 
                end_idx_main = -1 
        try:
            CN0_L1_ratio = CN0_L1_main[start_idx_main:end_idx_main] / CN0_L1_aux[start_idx_aux:end_idx_aux]
            CN0_L2_ratio = CN0_L2_main[start_idx_main:end_idx_main] / CN0_L2_aux[start_idx_aux:end_idx_aux]
            CN0_L5_ratio = CN0_L5_main[start_idx_main:end_idx_main] / CN0_L5_aux[start_idx_aux:end_idx_aux]
        except UnboundLocalError:
            CN0_L1_aux = ensure_same_length(CN0_L1_main,CN0_L1_aux,obs_aux.time.values)
            CN0_L2_aux = ensure_same_length(CN0_L2_main,CN0_L2_aux,obs_aux.time.values)
            CN0_L5_aux = ensure_same_length(CN0_L5_main,CN0_L5_aux,obs_aux.time.values) 

            CN0_L1_ratio = CN0_L1_main / CN0_L1_aux
            CN0_L2_ratio = CN0_L2_main / CN0_L2_aux
            CN0_L5_ratio = CN0_L5_main / CN0_L5_aux
        

    return CN0_L1_ratio, CN0_L2_ratio, CN0_L5_ratio

def angle_difference(angle1, angle2):
    return np.abs((angle1 - angle2 + 180) % 360 - 180)

def main():
    sv_list_brick = get_common_svlist(OBS_MAIN_BRICK,OBS_AUX_BRICK)
    sv_list_metal = get_common_svlist(OBS_MAIN_METAL,OBS_AUX_METAL)
    sv_list_glass = get_common_svlist(OBS_MAIN_GLASS,OBS_AUX_GLASS)
    sv_list_clear = get_common_svlist(OBS_MAIN_CLEAR,OBS_AUX_CLEAR)

    plot_ratios(sv_list_glass,'glass')
    plot_ratios(sv_list_metal,'metal')
    plot_ratios(sv_list_brick,'brick')
    plot_ratios(sv_list_clear,'')

    el_glass, az_glass = get_el_az('glass')
    el_metal, az_metal = get_el_az('metal')
    el_brick, az_brick = get_el_az('brick')
    el_clear, az_clear = get_el_az('')

    brick_sv = 'G11'
    glass_sv = 'G24'
    metal_sv = 'G28'
    clear_sv = 'G28'

    brick_CN0_L1_ratio, brick_CN0_L2_ratio, brick_CN0_L5_ratio = get_ratio_single_sv('brick',brick_sv)
    glass_CN0_L1_ratio, glass_CN0_L2_ratio, glass_CN0_L5_ratio = get_ratio_single_sv('glass',glass_sv)
    metal_CN0_L1_ratio, metal_CN0_L2_ratio, metal_CN0_L5_ratio = get_ratio_single_sv('metal',metal_sv)
    clear_CN0_L1_ratio, clear_CN0_L2_ratio, clear_CN0_L5_ratio = get_ratio_single_sv('',clear_sv)

    print('brick')
    print(np.nanmean(brick_CN0_L1_ratio))
    print(np.nanmean(brick_CN0_L2_ratio))
    print(np.nanmean(brick_CN0_L5_ratio))
    print('glass')
    print(np.nanmean(glass_CN0_L1_ratio))
    print(np.nanmean(glass_CN0_L2_ratio))
    print(np.nanmean(glass_CN0_L5_ratio))
    print('metal')
    print(np.nanmean(metal_CN0_L1_ratio))
    print(np.nanmean(metal_CN0_L2_ratio))
    print(np.nanmean(metal_CN0_L5_ratio))
    print('clear')
    print(np.nanmean(clear_CN0_L1_ratio))
    print(np.nanmean(clear_CN0_L2_ratio))
    print(np.nanmean(clear_CN0_L5_ratio))

    fig, ax = plt.subplots(3,1,figsize=(7,5),sharex=True,sharey=True)
    ax[0].set_title('L1')
    ax[0].plot(brick_CN0_L1_ratio, color='green', label=f'Brick RHP/LHP')
    ax[0].plot(glass_CN0_L1_ratio, color='blue', label=f'Glass RHP/LHP')
    ax[0].plot(metal_CN0_L1_ratio, color='orange', label=f'Metal RHP/LHP')
    ax[0].plot(clear_CN0_L1_ratio, color='purple', label=f'Clear RHP/LHP')
    ax[0].set_ylabel(r'$C/N_0$ Ratio')
    ax[0].legend()

    ax[1].set_title('L2')
    ax[1].plot(brick_CN0_L2_ratio, color='green', label=f'Brick RHP/LHP')
    ax[1].plot(glass_CN0_L2_ratio, color='blue', label=f'Glass RHP/LHP')
    ax[1].plot(metal_CN0_L2_ratio, color='orange', label=f'Metal RHP/LHP')
    ax[1].plot(clear_CN0_L2_ratio, color='purple', label=f'Clear RHP/LHP')
    ax[1].set_ylabel(r'$C/N_0$ Ratio')
    ax[1].legend()

    ax[2].set_title('L5')
    ax[2].plot(brick_CN0_L5_ratio, color='green', label=f'Brick RHP/LHP')
    ax[2].plot(glass_CN0_L5_ratio, color='blue', label=f'Glass RHP/LHP')
    ax[2].plot(metal_CN0_L5_ratio, color='orange', label=f'Metal RHP/LHP')
    ax[2].plot(clear_CN0_L5_ratio, color='purple', label=f'Clear RHP/LHP')
    ax[2].set_ylabel(r'$C/N_0$ Ratio')
    ax[2].set_xlabel(r'Time [s]')
    ax[2].set_xlim([-150, 5200])
    ax[2].legend()
    plt.tight_layout()
    plt.savefig(f'../figs/cn0_ratios_different_frequencies/brick_glass_metal_cn0_ratios_{str(np.datetime64("now"))}.pdf')
    plt.savefig(f'../figs/cn0_ratios_different_frequencies/brick_glass_metal_cn0_ratios_{str(np.datetime64("now"))}.png')
    plt.show()

if __name__ == '__main__':

    ## brick building with L5, 10m away
    # 3633416 is right-handed
    # 3632777 is left-handed
    path = "../Data/brick_wall/10m/090324/"
    OBS_MAIN_BRICK = gr.load(f"{path}brickwall_3633416_10m.obs",use="G")
    OBS_AUX_BRICK = gr.load(f"{path}brickwall_3632777_10m.obs",use="G")
    NAV_MAIN_BRICK = gr.load(f"{path}brickwall_3633416_10m_gps.nav")
    NAV_AUX_BRICK = gr.load(f"{path}brickwall_3632777_10m_gps.nav")
    PERP_WALL_BRICK = 106

    ## metal building with L5, 10m away
    # 3632777 is right-handed
    # 3633416 is left-handed
    path = "../Data/metal_wall/10m/290324/"
    OBS_MAIN_METAL = gr.load(f"{path}metalwall_3632777_10m.obs",use="G")
    OBS_AUX_METAL = gr.load(f"{path}metalwall_3633416_10m.obs",use="G")
    NAV_MAIN_METAL = gr.load(f"{path}metalwall_3632777_10m_gps.nav")
    NAV_AUX_METAL = gr.load(f"{path}metalwall_3633416_10m_gps.nav")
    PERP_WALL_METAL = 286

    ## glass building with L5, 10m away
    # 3632777 is right-handed
    # 3633416 is left-handed
    path = "../Data/glass_wall/10m/110424/"
    OBS_MAIN_GLASS = gr.load(f"{path}glasswall_3632777_10m.obs",use="G")
    OBS_AUX_GLASS = gr.load(f"{path}glasswall_3633416_10m.obs",use="G")
    NAV_MAIN_GLASS = gr.load(f"{path}glasswall_3632777_10m_gps.nav")
    NAV_AUX_GLASS = gr.load(f"{path}glasswall_3633416_10m_gps.nav")
    PERP_WALL_GLASS = 195

    ## clear sky with L5
    # 3632777 is right-handed
    # 3633416 is left-handed
    path = "../Data/clear_wall/170424/"
    OBS_MAIN_CLEAR = gr.load(f"{path}clear_3632777.obs",use="G") 
    OBS_AUX_CLEAR = gr.load(f"{path}clear_3633416.obs",use="G")
    NAV_MAIN_CLEAR = gr.load(f'{path}clear_3632777_gps.nav')
    NAV_AUX_CLEAR = gr.load(f'{path}clear_3633416_gps.nav')

    PLOT_RAW_CN0 = False
    PLOT_RATIO_SINGLE_SV = False

    main()