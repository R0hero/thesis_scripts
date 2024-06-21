import os 
from datetime import datetime
import subprocess
import re

def get_date():
    today = datetime.today()

    return today.strftime('%d%m%y')

def get_signal_band():
    while True:
        signal_band = input(f'Which signal are you working with?\n\
{" ".join([f"({i}) {band}" for i, band in enumerate(SIGNALS_AVAILABLE, start=1)])}\n')
        if signal_band.isdigit():
            signal_band = int(signal_band)
            if signal_band < 1 or signal_band > len(SIGNALS_AVAILABLE):
                print('Your choice is not valid.')
            else:
                return SIGNALS_AVAILABLE[signal_band-1]
        else:
            if not signal_band.upper() in SIGNALS_AVAILABLE:
                print('Your choice is not valid.')
            else:
                return signal_band.upper()

def select_material():
    materials_available = ['clean', 'metal', 'brick', 'glass']

    while True:
        material_input = input(f'Which material are you working with?\n\
{" ".join([f"({i}) {material}" for i, material in enumerate(materials_available, start=1)])}\n')
        if material_input.isdigit():
            material_input = int(material_input)
            if material_input < 1 or material_input > len(materials_available):
                print('Your choice is not valid.')
            else:
                return materials_available[material_input-1]
        else:
            if not material_input.lower() in materials_available:
                print('Your choice is not valid.')
            else:
                return material_input.lower()

def set_file_location(file, config_file, signal_band, single_frequency=True):
    if single_frequency:
        filename_string = f'SignalSource.filename='
        index = config_file.find(filename_string)
        config_file = config_file[:(index+len(filename_string))] + file + config_file[(index+len(filename_string)):]
        return config_file
    
    bands = split_signal_bands(signal_band)

    for i in range(len(bands)):
        filename_strings = f'SignalSource.filename{i}='
        index = config_file.find(filename_strings)
        config_file = config_file[:(index+len(filename_strings))] + file[i] + config_file[(index+len(filename_strings)):]
    return config_file

def select_signal_file(signal_band, material, single_frequency=True):
    path_to_files = f'{STORAGE_PATH}{material}{os.sep}'
    files = os.listdir(path_to_files)

    if single_frequency:
        file = [path_to_files+file for file in files if signal_band in file]
        return file[0]
    
    bands = split_signal_bands(signal_band)

    filtered_files = []
    for file in files:
        if any(band in file for band in bands):
            filtered_files.append(path_to_files+file)

    return filtered_files

def split_signal_bands(signal_band):
    pattern = r'L[0-9]'

    matches = re.findall(pattern, signal_band)

    return matches

def save_file(config_file):
    with open(SAVE_FILENAME, 'w') as file:
        file.write(config_file)

def predefined_chip_spacings():
    normal_spacings = (0.5, 0.15)
    tight_spacings = (0.2, 0.1)
    wide_spacings = (0.8, 0.3)
    
    while True:
        spacing_choice = input(f'Pick your predefined settings:\n\
(1): [normal] chip spacing: {normal_spacings[0]} chips, narrow chip spacing: {normal_spacings[1]} chips\n\
(2): [tight] chip spacing: {tight_spacings[0]} chips, narrow chip spacing: {tight_spacings[1]} chips\n\
(3): [wide] chip spacing: {wide_spacings[0]} chips, narrow chip spacings: {wide_spacings[1]} chips\n')
        if spacing_choice in ['1', '2', '3']:
            spacing_choice = int(spacing_choice)
            if spacing_choice == 1:
                return normal_spacings
            elif spacing_choice == 2:
                return tight_spacings
            elif spacing_choice == 3:
                return wide_spacings
        elif spacing_choice.lower() in ['normal', 'tight', 'wide']:
            if spacing_choice.lower() == 'normal':
                return normal_spacings
            elif spacing_choice.lower() == 'tight':
                return tight_spacings
            elif spacing_choice.lower() == 'wide':
                return wide_spacings
        else:
            print(f'Your choice is not valid.')

def is_decimal(input_str):
    try:
        float(input_str)
        return True
    except ValueError:
        return False

def define_chip_spacings():
    select_predefined_chip_spacings = input(f'Do you want to pick a predefined chip spacing configuration?\n\
(1) Yes (2) No\n')
    if select_predefined_chip_spacings == '1' or select_predefined_chip_spacings.lower() == 'y' or select_predefined_chip_spacings.lower() == 'yes':
        chip_spacings = predefined_chip_spacings()
        return chip_spacings
    
    chip_spacings_range = [0.2, 0.9]
    while True:
        chip_spacing_input = input(f'Please select a chip spacing in the range {chip_spacings_range[0]} to {chip_spacings_range[1]} chips\n')
        if is_decimal(chip_spacing_input):
            chip_spacing_input = float(chip_spacing_input)
            if chip_spacing_input >= chip_spacings_range[0] and chip_spacing_input <= chip_spacings_range[1]:
                break
        else:
            print(f'Your input is not valid.')
    
    narrow_chip_spacings_range = [0.1, chip_spacing_input]
    while True:
        narrow_chip_spacing_input = input(f'Please select a narrow chip spacing in the range {narrow_chip_spacings_range[0]} to {narrow_chip_spacings_range[1]}\n')
        if is_decimal(narrow_chip_spacing_input):
            narrow_chip_spacing_input = float(narrow_chip_spacing_input)
            if narrow_chip_spacing_input >= narrow_chip_spacings_range[0] and narrow_chip_spacing_input <= narrow_chip_spacings_range[1]:
                break
        else:
            print(f'Your input is not valid.')
    return (chip_spacing_input, narrow_chip_spacing_input)

def set_output_path(config_file, signal_band, material, chip_spacings):
    rinex_output_path_string = f'PVT.rinex_output_path='
    rinex_name_path_string = f'PVT.rinex_name='
    kml_output_path_string = f'PVT.kml_output_path='
    
    rinex_output = f'data{os.sep}{signal_band}{os.sep}{material}{os.sep}chip_{chip_spacings[0]}narrow_{chip_spacings[1]}{os.sep}'
    kml_output = f'data{os.sep}{signal_band}{os.sep}{material}{os.sep}chip_{chip_spacings[0]}narrow_{chip_spacings[1]}{os.sep}kml{os.sep}'
    rinex_name = f'{get_date()}'

    rinex_output_index = config_file.find(rinex_output_path_string)
    config_file = config_file[:(rinex_output_index+len(rinex_output_path_string))] + rinex_output + config_file[(rinex_output_index+len(rinex_output_path_string)):]
    
    rinex_name_index = config_file.find(rinex_name_path_string)
    config_file = config_file[:(rinex_name_index+len(rinex_name_path_string))] + rinex_name + config_file[(rinex_name_index+len(rinex_name_path_string)):]
    
    kml_index = config_file.find(kml_output_path_string)
    config_file = config_file[:(kml_index+len(kml_output_path_string))] + kml_output + config_file[(kml_index+len(kml_output_path_string)):]
    
    return config_file

def set_chip_spacings(config_file, signal_band, chip_spacings, single_frequency=True):
    if single_frequency:
        if signal_band == 'L1':
            signal_band = '1C'
        chip_spacing_string = f'Tracking_{signal_band}.early_late_space_chips='
        narrow_chip_spacing_string = f'Tracking_{signal_band}.early_late_space_narrow_chips='

        chip_spacing_index = config_file.find(chip_spacing_string)
        config_file = config_file[:(chip_spacing_index+len(chip_spacing_string))] + str(chip_spacings[0]) + config_file[(chip_spacing_index+len(chip_spacing_string)):]

        narrow_chip_spacing_index = config_file.find(narrow_chip_spacing_string)
        config_file = config_file[:(narrow_chip_spacing_index+len(narrow_chip_spacing_string))] + str(chip_spacings[1]) + config_file[(narrow_chip_spacing_index+len(narrow_chip_spacing_string)):]

        return config_file
    
    bands = split_signal_bands(signal_band)
    for _, signal_band in enumerate(bands):
        if signal_band == 'L1':
            signal_band = '1C'
        chip_spacing_string = f'Tracking_{signal_band}.early_late_space_chips='
        narrow_chip_spacing_string = f'Tracking_{signal_band}.early_late_space_narrow_chips='

        chip_spacing_index = config_file.find(chip_spacing_string)
        config_file = config_file[:(chip_spacing_index+len(chip_spacing_string))] + str(chip_spacings[0]) + config_file[(chip_spacing_index+len(chip_spacing_string)):]

        narrow_chip_spacing_index = config_file.find(narrow_chip_spacing_string)
        config_file = config_file[:(narrow_chip_spacing_index+len(narrow_chip_spacing_string))] + str(chip_spacings[1]) + config_file[(narrow_chip_spacing_index+len(narrow_chip_spacing_string)):]
    return config_file

def check_to_run_receiver():
    run_gnsssdr_input = input(f'Do you wish to run GNSS-SDR with this file?\n\
(1) Yes (2) No\n')
    if run_gnsssdr_input == '1' or run_gnsssdr_input.lower() == 'y' or run_gnsssdr_input.lower == 'yes':
        start_receiver_command = f'gnss-sdr --config_file={SAVE_FILENAME}'
        subprocess.run(start_receiver_command, shell=True)

def check_if_single_frequency(signal_band):
    if 'L1' in signal_band and 'L5' in signal_band:
        return False
    return True

def main():
    signal_band = get_signal_band()
    file_name = f'teleorbit_{signal_band}_wsl_template.conf'

    single_frequency = check_if_single_frequency(signal_band)

    material = select_material()
    signal_file = select_signal_file(signal_band, material, single_frequency)
    chip_spacings = define_chip_spacings()

    with open(FILE_PATH+file_name) as file:
        config_file = file.read()
    
    config_file = set_file_location(signal_file, config_file, signal_band, single_frequency)
    config_file = set_chip_spacings(config_file, signal_band, chip_spacings, single_frequency)
    config_file = set_output_path(config_file, signal_band, material, chip_spacings)

    save_file(config_file)

    check_to_run_receiver()



if __name__ == '__main__':
    SIGNALS_AVAILABLE = ['L1', 'L5', 'L1+L5']
    FILE_PATH = f'conf{os.sep}templates{os.sep}'
    STORAGE_PATH = f'{os.sep}mnt{os.sep}f{os.sep}'
    SAVE_FILENAME = f'config.conf'

    main()