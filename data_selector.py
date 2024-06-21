import os
import csv 
import re
import glob
import numpy as np

def get_materials():

    contents = os.listdir(PATH_TO_DATA)

    folders = [item for item in contents if os.path.isdir(os.path.join(PATH_TO_DATA, item)) and "_wall" in item]

    folders_ = [folder[:-5] for folder in folders]
    return folders_

def pick_material(materials_to_pick):
    while True:
        material_input = input(f'Which material do you wish to pick? Enter the material name or its corresponding number:\
                               \n{" ".join([f"({i}) {material}" for i, material in enumerate(materials_to_pick, start=1)])}\n')

        if material_input.isdigit():
            material_index = int(material_input)
            if 1 <= material_index <= len(materials_to_pick):
                selected_material = materials_to_pick[material_index - 1]
                print(f'You selected: {selected_material}')
                return selected_material
            else:
                print('Invalid material number.')
        else:
            if material_input in materials_to_pick:
                print(f'You selected: {material_input}')
                return material_input
            else:
                print('Invalid material name.')

def get_distances(material):
    contents = os.listdir(PATH_TO_DATA+material+'_wall'+os.sep)
    return contents

def pick_distances(distances_to_pick):
    while True:
        distance_input = input(f'Which distance do you wish to pick? Enter the distance or its corresponding number:\
                               \n{" ".join([f"({i}) {material}" for i, material in enumerate(distances_to_pick, start=1)])}\n')

        if distance_input.isdigit():
            distance_index = int(distance_input)
            if 1 <= distance_index <= len(distances_to_pick):
                selected_distance = distances_to_pick[distance_index - 1]
                print(f'You selected: {selected_distance}')
                return selected_distance
            else:
                print('Invalid distance number.')
        else:
            if distance_input in distances_to_pick:
                print(f'You selected: {distance_input}')
                return distance_input
            else:
                print('Invalid distance name.')

def get_dates(material,distance):
    contents = os.listdir(PATH_TO_DATA+material+'_wall'+os.sep+distance+os.sep)
    return contents

def pick_date(dates_to_pick):
    if len(dates_to_pick) == 1:
        print(f'Only one date available. {"".join(dates_to_pick)} has been picked.')
        return "".join(dates_to_pick)
    
    while True:
        date_input = input(f'You currently have {len(dates_to_pick)} dates available to pick from:\
                           \n{" ".join([f"({i}) {date}" for i, date in enumerate(dates_to_pick, start=1)])}\n')

        if date_input.isdigit() and len(date_input)<3:
            date_index = int(date_input)
            if 1 <= date_index <= len(dates_to_pick):
                selected_date = dates_to_pick[date_index - 1]
                print(f'You selected: {selected_date}')
                return selected_date
            else:
                print('Invalid date number.')
        else:
            if date_input in dates_to_pick:
                print(f'You selected: {date_input}')
                return date_input
            else:
                print('Invalid date.')

def check_if_polarized_antenna(picked_path):
    serial_numbers = set()

    for filename in os.listdir(picked_path):
        matches = re.findall(r'\d{7}', filename)

        serial_numbers.update(matches)

    return list(serial_numbers)

def read_polarization_file(picked_path):
    with open(picked_path+POLARIZATION_FILE) as file:
        csv_reader = csv.reader(file)

        data = []
        for row in csv_reader:
            data.append(row)
        return data[1:]

def create_polarization_file(picked_path,serial_numbers):
    while True:
        polarization_input = input(f'A polarization details file was not found in the folder.\
                                   \nWhich of the following serial numbers are the right-hand polarized (RHP) antenna?\
                                   \n{" ".join([f"({i}) {serial_number}" for i, serial_number in enumerate(serial_numbers, start=1)])}\n')
        if polarization_input.isdigit():
            polarization_index = int(polarization_input)
            if 1 <= polarization_index <= len(serial_numbers):
                selected_serial_number = serial_numbers[polarization_index - 1]
                break
            else:
                print('Invalid number.')
        else:
            if polarization_input in serial_numbers:
                selected_serial_number = polarization_input
                break
            else:
                print('Invalid serial number.')
    
    if serial_numbers.index(selected_serial_number) == 0:
        other_serial_number = serial_numbers[1]
    else:
        other_serial_number = serial_numbers[0]

    content = [
        ['serial number', 'polarization'],
        [selected_serial_number, 'RHP'],
        [other_serial_number, 'LHP']
    ]

    with open(picked_path+POLARIZATION_FILE, 'w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerows(content)
    print('Polarization file has been created.')

def check_polarization_file(picked_path,serial_numbers):
    if os.path.exists(picked_path+POLARIZATION_FILE):
        print('Polarization file found!')
        data = read_polarization_file(picked_path)
    else:
        create_polarization_file(picked_path,serial_numbers)
        data = read_polarization_file(picked_path)
    return data   

def search_files(files,serial_number):
    for file_path in files:
        if serial_number in os.path.basename(file_path):
            return file_path


def get_main_aux_files(picked_path,RHP_serial_number,LHP_serial_number,constellation):

    obs_files = glob.glob(os.path.join(picked_path, '*.obs'))
    nav_files = glob.glob(os.path.join(picked_path, f'*{constellation.lower()}.nav'))

    obs_main = search_files(obs_files,RHP_serial_number)
    obs_aux = search_files(obs_files,LHP_serial_number)
    nav_main = search_files(nav_files,RHP_serial_number)
    nav_aux = search_files(nav_files,LHP_serial_number)
    return obs_main, nav_main, obs_aux, nav_aux

def get_file_paths(data,serial_numbers, picked_path, constellation):
    if serial_numbers[0] in data[0]:
        if data[0][1] == 'LHP':
            print(f'Loading {serial_numbers[0]} as LHP')
            print(f'Loading {serial_numbers[1]} as RHP')
            obs_main, nav_main, obs_aux, nav_aux = get_main_aux_files(picked_path,serial_numbers[1],serial_numbers[0],constellation)
        else:
            print(f'Loading {serial_numbers[0]} as RHP')
            print(f'Loading {serial_numbers[1]} as LHP')
            obs_main, nav_main, obs_aux, nav_aux = get_main_aux_files(picked_path,serial_numbers[0],serial_numbers[1],constellation)
    else:
        if data[1][1] == 'LHP':
            print(f'Loading {serial_numbers[1]} as RHP')
            print(f'Loading {serial_numbers[0]} as LHP')
            obs_main, nav_main, obs_aux, nav_aux = get_main_aux_files(picked_path,serial_numbers[1],serial_numbers[0],constellation)
        else:
            print(f'Loading {serial_numbers[0]} as RHP')
            print(f'Loading {serial_numbers[1]} as LHP')
            obs_main, nav_main, obs_aux, nav_aux = get_main_aux_files(picked_path,serial_numbers[0],serial_numbers[1],constellation)
    
    return obs_main, nav_main, obs_aux, nav_aux

def custom_sort(file):
    distances = ['_2.5m', '_5m', '_10m']
    for distance in distances:
        if distance in file:
            return distances.index(distance)
    return len(distances)  

def get_distances_per_file(picked_path, serial_numbers, constellation):
    distances = ['_2.5m', '_5m', '_10m']

    obs_filepaths = []
    nav_filepaths = []

    obs_files = glob.glob(os.path.join(picked_path, '*.obs'))
    nav_files = glob.glob(os.path.join(picked_path, f'*{constellation.lower()}.nav'))
    
    obs_files.sort(key=custom_sort)
    nav_files.sort(key=custom_sort)
    
    for file in obs_files:
        if any(serial in file for serial in serial_numbers):
            if any(distance in file for distance in distances):
                obs_filepaths.append(file)
    
    for file in nav_files:
        if any(serial in file for serial in serial_numbers):
            if any(distance in file for distance in distances):
                nav_filepaths.append(file)
    
    return obs_filepaths[0], nav_filepaths[0], obs_filepaths[1], nav_filepaths[1], obs_filepaths[2], nav_filepaths[2] 

def pick_constellation():
    options = ['GPS', 'Galileo']

    while True:
        constellation_pick = input(f'Which constellation are you working with?\n{" ".join([f"({i}) {value}" for i, value in enumerate(options, start=1)])}\n')

        if constellation_pick.isdigit():
            constellation_index = int(constellation_pick)
            if 1 <= constellation_index <= len(options):
                constellation = options[constellation_index - 1]
                print(f'You picked {constellation}.')
                return constellation
            else:
                print('Your option is not within the range.')
        else:
            if constellation_pick.lower() in [option.lower() for option in options]:
                constellation_index = np.where([constellation_pick.lower() == option.lower() for option in options])[0][0]
                constellation = options[constellation_index]
                print(f'You picked {constellation}')
                return constellation
            else:
                print(f'You pick an invalid constellation.')

def data_selector():
    materials_to_pick = get_materials()
    material = pick_material(materials_to_pick)

    if not material == 'clear':
        distances_to_pick = get_distances(material)
        distance = pick_distances(distances_to_pick)
    else:
        distance = ''
    
    dates_to_pick = get_dates(material,distance)
    date = pick_date(dates_to_pick)

    picked_path = f'{PATH_TO_DATA}{material}_wall{os.sep}{distance}{os.sep}{date}{os.sep}'

    serial_numbers = check_if_polarized_antenna(picked_path)

    constellation = pick_constellation()

    if len(serial_numbers) == 2:
        print(f'There was found 2 different serial numbers in the folder. Checking polarization.')
        data = check_polarization_file(picked_path,serial_numbers)
        obs_main, nav_main, obs_aux, nav_aux = get_file_paths(data, serial_numbers, picked_path, constellation)
        return obs_main, nav_main, obs_aux, nav_aux
    if len(serial_numbers) == 3:
        print(f'There was found 3 different serial numbers in the folder. This is a distance correlation folder.')
        obs2dot5m, nav2dot5m, obs5m, nav5m, obs10m, nav10m = get_distances_per_file(picked_path, serial_numbers, constellation)
        return obs2dot5m, nav2dot5m, obs5m, nav5m, obs10m, nav10m
        
PATH_TO_DATA = f'..{os.sep}Data{os.sep}'
POLARIZATION_FILE = f'polarization_details.csv'

if __name__ == "__main__":
    data_selector()