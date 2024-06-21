# own functions
from data_selector import data_selector
from sv_functions import *

import georinex as gr
import matplotlib.pyplot as plt
plt.ion()
from matplotlib import cm
import re
import numpy as np
from scipy.spatial import ConvexHull

def get_material_and_angle(file):
    if 'glass' in file:
        wall_type = 'glass'
        perp_from_wall = 195 # deg
    elif 'metal' in file:
        wall_type = 'metal'
        perp_from_wall = 286 # deg
    elif 'brick' in file:
        wall_type = 'brick'
        perp_from_wall = 106 # deg
    elif 'clear' in file:
        wall_type = ''
        perp_from_wall = None # deg
    return wall_type, perp_from_wall

def get_distance_and_date(file):
    pattern_date = r'\d{6}'
    pattern_distance = r'(?:\.|\_|\b)(?:10m|5m|2\.5m)(?:\.|\_|\b)'
    match_date = re.search(pattern_date, file)[0]

    if not 'clear' in file:
        match_distance = re.search(pattern_distance, file)[0]
        if match_distance[0] == '_':
            match_distance = match_distance[1:]
        if match_distance[-1] == '.':
            match_distance = match_distance[:-1]
        return match_distance, match_date
    
    return '', match_date

def cart2pol(x, y):
    """
    source: https://stackoverflow.com/questions/20924085/python-conversion-between-coordinates
    """
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def plot_circle_on_polar(r):
    theta = np.linspace(0, 2*np.pi, 100) 
    x = r * np.cos(theta)  
    y = r * np.sin(theta)  

    rho, theta = cart2pol(x,y)
    plt.polar(theta, rho)  # Plot the circle on a polar plot

def get_azimuth_elevation(obs, nav, galileo_nav=False, galileo_system=None):
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
                sat_positions, _, _ = calcSatPos(nav, t[t_i], sv_list, galileo_nav=galileo_nav, galileo_system=galileo_system)
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
    return az, el, sv_list

def select_sv_with_and_without_L5(obs, sv_list):
    sv_without_L5 = np.sum(np.isnan(obs.sel(sv=sv_list)['S5Q'].values),axis=0) == len(obs['S5Q'].values)

    sv_with_L5 = np.nonzero(~sv_without_L5)[0]
    sv_without_L5 = np.nonzero(sv_without_L5)[0]

    return sv_with_L5, sv_without_L5

def Brewster_elevation():
    while True:
        draw_prompt = input('Do you wish to plot the Brewster elevation angle for the material?\n(1) yes (2) no\n')
        if draw_prompt.isdigit():
            if draw_prompt == '1':
                return True
            elif draw_prompt == '2':
                return False
            else:
                print('The number is not in the given range. Please pick a valid number')
        elif draw_prompt.lower() == 'yes' or draw_prompt.lower() == 'y':
                return True
        elif draw_prompt.lower() == 'no' or draw_prompt.lower() == 'n':
                return False
        else:
            print('The input is not valid.')

def calculate_circle(center_x, center_y, radius=5, num_points=8):
    theta = np.linspace(0, 2*np.pi, num_points)
    x = center_x + radius * np.cos(theta)
    y = center_y + radius * np.sin(theta)
    return x, y

def get_edges_of_polygon(theta, r, radius=5, num_points=8):
    circle_x = np.array([])
    circle_y = np.array([])
    for i in range(len(theta)):
        circle_x_calc, circle_y_calc = calculate_circle(theta[i], r[i], radius=radius, num_points=num_points)
        circle_x = np.append(circle_x,circle_x_calc)
        circle_y = np.append(circle_y,circle_y_calc)

    all_points = np.column_stack((circle_x.flatten(), circle_y.flatten()))

    hull = ConvexHull(all_points)
    edge_idx = hull.vertices
    edge_points = all_points[edge_idx]
    edges_x = edge_points[:,0]
    edges_y = edge_points[:,1]

    return edges_x, edges_y

def plot_skyplot(az, el, sv_with_L5, sv_without_L5, sv_list, title_string, perp_from_wall, material, date, galileo_system=False):

    if galileo_system:
        constellation = 'Galileo'
    else:
        constellation = 'GPS'

    if galileo_system:
        col = 'purple'
    else:
        col_with_L5 = 'darkgreen'
        col_without_L5 = 'darkorange'

    plt.figure(figsize=(5,5))
    if galileo_system:
        plt.polar(np.deg2rad(az.T),el.T, linewidth=3, color=col)
    else:
        plt.polar(np.deg2rad(az[sv_with_L5,:].T),el[sv_with_L5,:].T, linewidth=3, color=col_with_L5)
        plt.polar(np.deg2rad(az[sv_without_L5,:].T),el[sv_without_L5,:].T,linewidth=3, color=col_without_L5)
    
    if perp_from_wall == None and not galileo_system:
        legend_handles = [
            plt.Line2D([], [], color=col_with_L5, linewidth=2, label='SV with L5'),
            plt.Line2D([], [], color=col_without_L5, linewidth=2, label='SV without L5')
        ]
    elif not galileo_system:
        legend_handles = [
            plt.Line2D([], [], color=col_with_L5, linewidth=2, label='SV with L5'),
            plt.Line2D([], [], color=col_without_L5, linewidth=2, label='SV without L5'),
            plt.Line2D([], [], color='red', linewidth=0.5, linestyle='--', label='Perp. with wall'),
            plt.Line2D([], [], color='black', linewidth=1.5, linestyle='--', label='Wall placement')
        ]
    elif not perp_from_wall == None and galileo_system:
        legend_handles = [
            plt.Line2D([], [], color=col, linewidth=2, label='SV'),
            plt.Line2D([], [], color='red', linewidth=0.5, linestyle='--', label='Perp. with wall'),
            plt.Line2D([], [], color='black', linewidth=1.5, linestyle='--', label='Wall placement')
        ]
    else:
        legend_handles = [
            plt.Line2D([], [], color=col, linewidth=2, label='SV')
        ]


    # set theta direction and zero location
    plt.gca().set_theta_direction(-1)
    plt.gca().set_theta_zero_location('N')  

    ticks = np.arange(90, 0, -15)
    tick_labels = [f"{tick}°" for tick in ticks] 
    plt.gca().set_rticks(ticks)
    plt.gca().set_yticklabels(tick_labels)  
    plt.gca().set_ylim(90,0)  

    idx = np.zeros(np.shape(az)[0],dtype=np.int64)
    for i, _ in enumerate(sv_list):
        nan_mask = np.isnan(az[i,:])
        idx[i] = len(az[i,:]) - np.argmax(np.flipud(~nan_mask)) - 1
    sv_list_array = np.array(sv_list)
    for i, sv in enumerate(sv_list):
        plt.annotate(sv_list[i], xy=(np.deg2rad(az[i, idx[i]]), el[i, idx[i]]),
                    xytext=(1, 5), textcoords='offset points')
        
        if galileo_system:
            plt.scatter(np.deg2rad(az[i, idx[i]]), el[i, idx[i]],
                      marker='o', s=35, color=col)
        else:
            if sv in sv_list_array[sv_without_L5]:
                plt.scatter(np.deg2rad(az[i, idx[i]]), el[i, idx[i]],
                        marker='o', s=35, color=col_without_L5)
            else:
                plt.scatter(np.deg2rad(az[i, idx[i]]), el[i, idx[i]],
                        marker='o', s=35, color=col_with_L5)
            
    plt.title(title_string)

    if not perp_from_wall == None:
        plt.polar([np.deg2rad(perp_from_wall-180),np.deg2rad(perp_from_wall)],[85,0],color='red',linewidth=1.5,linestyle='--',label=f'Perp. with wall')
        plt.polar([np.deg2rad(perp_from_wall-90-15), np.deg2rad(perp_from_wall+90+15)],[70, 70],color='black',linewidth=1.5,linestyle='--',label=f'Wall placement')
    plt.legend(handles=legend_handles,loc='upper right', bbox_to_anchor=(0.55, 0., 0.59, 1.16))
    
    savepath = f'../figs/skyplots/'
    if material == '':
        save_filename = f'{date}_{constellation}'
    else:
        save_filename = f'{material}_10m_{date}_{constellation}'
    if Brewster_elevation():
        if material == 'glass':
            r = 90 - 20.66
        elif material == 'metal':
            r = 90 - 13.94
        elif material == 'brick':
            r = 90 - 27.25
        plot_circle_on_polar(r)

        save_filename += f'_with_Brewster'
    plt.savefig(f'{savepath}{save_filename}.png')
    plt.savefig(f'{savepath}{save_filename}.pdf')
    plt.show()

    while True:
        mark_sv = input(f'{" ".join([f"({i}) {sv}" for i, sv in enumerate(sv_list,start=1)])})\nDo you want to mark a satellite on the skyplot?\n')
        if mark_sv.lower() == 'n' or mark_sv.lower() == 'no':
            return
        if mark_sv.isdigit():
            input_sv = int(mark_sv)
            input_sv = sv_list[input_sv - 1]

            input_sv_index = sv_list.index(input_sv)
            non_nan_idx = ~np.isnan(az[input_sv_index, :])
            edges_x, edges_y = get_edges_of_polygon(az[input_sv_index, non_nan_idx], el[input_sv_index, non_nan_idx], radius=5, num_points=20)

            plt.fill(np.deg2rad(edges_x), edges_y, color='blue', alpha=0.35)
            save_filename += f'_{input_sv}_focused'
            break
        elif mark_sv in sv_list_array:
            input_sv_index = sv_list.index(mark_sv)
            non_nan_idx = ~np.isnan(az[input_sv_index, :])
            edges_x, edges_y = get_edges_of_polygon(az[input_sv_index, non_nan_idx], el[input_sv_index, non_nan_idx], radius=5, num_points=20)

            plt.fill(np.deg2rad(edges_x), edges_y, color='blue', alpha=0.35)
            save_filename += f'_{mark_sv}_focused'
            break
        else:
            print(f'Input not valid.')
        
    plt.savefig(f'{savepath}{save_filename}.png')
    plt.savefig(f'{savepath}{save_filename}.pdf')


    input("Press Enter to close the plot window...")

def main():
    files = data_selector()

    if len(files) == 4:
        obs, nav, _, _ = files
    elif len(files) == 6:
        _, _, _, _, obs, nav = files

    material, perp_from_wall = get_material_and_angle(obs)
    distance, date = get_distance_and_date(obs)

    if not material == '':
        title_string = f'{distance[:-1]} m to {material} wall'
    else:
        title_string = f'clear sky conditions'

    if 'galileo' in nav:
        obs = gr.load(obs,use="E")
        galileo_nav = True
        galileo_system = 'E5A'
    else:
        obs = gr.load(obs,use="G")
        galileo_nav = False
        galileo_system = None
    nav = gr.load(nav)

    az, el, sv_list = get_azimuth_elevation(obs, nav, galileo_nav=galileo_nav, galileo_system=galileo_system)

    sv_with_L5, sv_without_L5 = select_sv_with_and_without_L5(obs, sv_list)

    plot_skyplot(az, el, sv_with_L5, sv_without_L5, sv_list, title_string, perp_from_wall, material, date, galileo_system=galileo_system)

if __name__ == '__main__':
    main()