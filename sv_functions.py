import numpy as np
from pyproj import Transformer

def correctTime(time):
    """
    correct time accounting for beginning or end of week crossover.

    Input:
        time : time in seconds
    Output:
        corr_time : corrected time in seconds
    
    Kai Borre 04-01-96
    Copyright by Kai Borre

    Translated to python January 26th 2024
    Benjamin Elsholm
    """

    half_week = 302400 # seconds

    corr_time = time
    if not np.shape(time) == ():
        idx_lower = np.where(time > half_week)
        idx_upper = np.where(time > -half_week)
        corr_time[idx_lower] = time[idx_lower] - 2*half_week
        corr_time[idx_upper] = time[idx_upper] + 2*half_week
    else:
        if time > half_week:
            corr_time = time - 2*half_week
        elif time < -half_week:
            corr_time = time + 2*half_week

    return corr_time

def ecef_to_lla(x, y, z, ellps='EPSG:4978', datum='EPSG:4326'):
    """
    Convert ECEF (Earth-Centered, Earth-Fixed) coordinates to geodetic latitude, longitude, and altitude (LLA) using the WGS84 datum as default.

    Input:
        x, y, z : ECEF coordinates in meters
        ellps : ellipsoidal model to use, default is WGS84
        datum : datum model to use, default is WGS84

    Output:
        lat : geodetic latitude in degrees
        lon : geodetic longitude in degrees 
        alt : geodetic altitude in meters
    """
    # Define the ECEF and LLA coordinate system
    transformer = Transformer.from_crs(f"{ellps}", f"{datum}")

    # Convert ECEF coordinates to LLA
    lat, lon, alt = transformer.transform(x, y, z)

    return lat, lon, alt

def datetime_to_gpsweek(dt_time, gps_epoch=np.datetime64('1980-01-06')):
    """
    function to convert np.datetime64 time to a GPS week and second of the GPS week based on the first epoch of GPS.

    input:
        dt_time : time in np.datetime64
        gps_epoch (optional) : the first epoch of GPS
    output:
        gps_week : number of weeks since first epoch of GPS
        seconds_of_week : number of seconds in the GPS week
    """
    seconds_in_day = 24*60*60
    seconds_in_week = seconds_in_day*7

    delta = dt_time - gps_epoch
    delta /= np.timedelta64(1,'s')

    gps_week = int(delta // seconds_in_week)

    seconds_of_week = int(delta % seconds_in_week)

    return gps_week, seconds_of_week

def calcSatPos(nav, t, sv_list, use_iono_free_correction=False, galileo_nav=False, galileo_system=None):
    """
    Satellite position, velocity and relativistic clock correction for satelites.

    Input:
        nav.sel(sv=sv) : Navigation struct (from geoRinex load function)
        t : transmission time
        sv_list : list of satellites to be processed

    Output:
        sat_positions : position of satellites (in ECEF) 
        sat_velocity : velocity of satellites
        sat_clock_relativistic_correction : relativistic correction of satellite clocks

    Kai Borre 04-09-96
    Copyright by Kai Borre
    Updated by Darius Plausinaitis, Peter Rinder and Nicolaj Bertelsen

    Modified and extended by Daniel Olesen, DTU Space 2016

    Translated to python January 26th 2024
    Benjamin Elsholm
    """
    
    ## variables
    Omega_dot = 7.2921151467e-5 # Earth rotation rate [rad/s]
    GM = 3.986005e14            # Universal gravitational constant times the mass of the Earth [m^3/s^2]
    F = -4.442807633e-10        # Constant [sec/(meter)^(1/2)]
    gpsPi = 3.1415926535898

    # check if only a single satellite in sv_list
    if np.shape(sv_list) == ():
        sv_list = [sv_list]
    
    num_sv = len(sv_list)
    ## initializing matrices
    sat_clock_correction = np.zeros(num_sv,dtype=np.float64)
    sat_clock_relativistic_correction = np.zeros(num_sv,dtype=np.float64)
    sat_positions = np.zeros((3, num_sv),dtype=np.float64)
    sat_velocity = np.zeros((3, num_sv),dtype=np.float64)
    ## computing
    for sat_no, sv in enumerate(sv_list):
        idx = np.where(~np.isnan(nav.sel(sv=sv).Crs.values))[0][0]

        # find time difference
        dt = correctTime(t - nav.sel(sv=sv).TransTime.values[idx])
        
        # calculate clock correction
        if use_iono_free_correction:
            sat_clock_correction = nav.sel(sv=sv).SVclockDriftRate.values[idx] * dt + nav.sel(sv=sv).SVclockDrift.values[idx] * dt + nav.sel(sv=sv).SVclockBias.values[idx]
        elif galileo_nav and galileo_system == 'E5A':
            sat_clock_correction = nav.sel(sv=sv).SVclockDriftRate.values[idx] * dt + nav.sel(sv=sv).SVclockDrift.values[idx] * dt + nav.sel(sv=sv).SVclockBias.values[idx] - nav.sel(sv=sv).BGDe5a.values[idx]
        elif galileo_nav and galileo_system == 'E5B':
            sat_clock_correction = nav.sel(sv=sv).SVclockDriftRate.values[idx] * dt + nav.sel(sv=sv).SVclockDrift.values[idx] * dt + nav.sel(sv=sv).SVclockBias.values[idx] - nav.sel(sv=sv).BGDe5b.values[idx]
        else:
            sat_clock_correction = nav.sel(sv=sv).SVclockDriftRate.values[idx] * dt + nav.sel(sv=sv).SVclockDrift.values[idx] * dt + nav.sel(sv=sv).SVclockBias.values[idx] - nav.sel(sv=sv).TGD.values[idx]
        time = t - sat_clock_correction

        # restore semi-major axis
        a = nav.sel(sv=sv).sqrtA.values[idx]**2

        # time correction
        tk = correctTime(time - nav.sel(sv=sv).Toe.values[idx])

        # initial mean motion
        n0 = np.sqrt(GM / a**3)
        # mean motion
        n = n0 + nav.sel(sv=sv).DeltaN.values[idx]
        
        # mean anomaly
        M = nav.sel(sv=sv).M0.values[idx] + n*tk
        # derivative of mean anomaly
        MDot = n
        # reduce mean anomaly to between 0 and 360 deg
        M = newMod(M + 2*gpsPi, 2*gpsPi)
        
        # initial guess eccentric anomaly
        E = M
        iter = 0
        iter_max = 10 # 
        while True:
            # update eccentric anomaly
            E_old = E
            E = M + nav.sel(sv=sv).Eccentricity.values[idx]*np.sin(E)
            # find difference
            dE = newMod(E-E_old, 2*gpsPi)
            # check if low enough to break
            if abs(dE).all() < 1e-12 or iter == iter_max:
                break
            iter += 1
        # find derivative of eccentric anomaly
        EDot = MDot / (1.0 - nav.sel(sv=sv).Eccentricity.values[idx]*np.cos(E))
        
        # reduce eccentric anomaly to between 0 and 360 deg
        E = newMod(E + 2*gpsPi, 2*gpsPi)
        
        # compute relativistic correction term
        dtr = F * nav.sel(sv=sv).Eccentricity.values[idx] * nav.sel(sv=sv).sqrtA.values[idx] * np.sin(E)
        
        # calculate true anomaly
        nu = np.arctan2(np.sqrt(1 - nav.sel(sv=sv).Eccentricity.values[idx]**2)*np.sin(E), np.cos(E)-nav.sel(sv=sv).Eccentricity.values[idx])
        
        # calculate derivative of true anomaly
        nuDot = np.sin(E) * EDot * (1.0 + nav.sel(sv=sv).Eccentricity.values[idx]*np.cos(nu)) / (np.sin(nu)*(1-nav.sel(sv=sv).Eccentricity.values[idx]*np.cos(E)))
        
        # compute angle phi (latitude)
        phi = nu + nav.sel(sv=sv).omega.values[idx]
        # reduce latitude to between 0 and 360 deg
        phi = newMod(phi,2*gpsPi)
        
        # calculate corrections for latitude, radius and inclination
        corr_u = nav.sel(sv=sv).Cus.values[idx]*np.sin(2.0*phi) + nav.sel(sv=sv).Cuc.values[idx]*np.cos(2.0*phi)
        corr_r = nav.sel(sv=sv).Crs.values[idx]*np.sin(2.0*phi) + nav.sel(sv=sv).Crc.values[idx]*np.cos(2.0*phi)
        corr_i = nav.sel(sv=sv).Cis.values[idx]*np.sin(2.0*phi) + nav.sel(sv=sv).Cic.values[idx]*np.cos(2.0*phi)
        # calculate latitude, radius and inclination
        u = phi + corr_u                                                                            # latitude
        r = a * (1 - nav.sel(sv=sv).Eccentricity.values[idx]*np.cos(E)) + corr_r                       # radius
        i = nav.sel(sv=sv).Io.values[idx] + nav.sel(sv=sv).IDOT.values[idx] * tk + corr_i                 # inclination
        
        # calculate derivatives of latitude, radius and inclination
        uDot = nuDot + 2 * (nav.sel(sv=sv).Cus.values[idx]*np.cos(2*u) - nav.sel(sv=sv).Cuc.values[idx]*np.sin(2*u)) * nuDot
        rDot = (a*nav.sel(sv=sv).Eccentricity.values[idx]*np.sin(E)*n)/(1-nav.sel(sv=sv).Eccentricity.values[idx]*np.cos(E)) + 2*(nav.sel(sv=sv).Crs.values[idx]*np.cos(2*u)-nav.sel(sv=sv).Crc.values[idx]*np.sin(2*u))*nuDot
        iDot = nav.sel(sv=sv).IDOT.values[idx] + (nav.sel(sv=sv).Cis.values[idx]*np.cos(2*u) - nav.sel(sv=sv).Cic.values[idx]*np.sin(2*u))*2*nuDot

        # compute angle between ascending node and the Greenwich meridian
        Omega = nav.sel(sv=sv).Omega0.values[idx] + (nav.sel(sv=sv).OmegaDot.values[idx] - Omega_dot)*tk - Omega_dot * nav.sel(sv=sv).Toe.values[idx]

        # reduce Omega to between 0 and 360 deg
        Omega = newMod(Omega + 2*gpsPi, 2*gpsPi)
        
        # compute derivative of the angle 
        OmegaDot = nav.sel(sv=sv).OmegaDot.values[idx] - Omega_dot

        # define intermediate calculations for simplicity
        xpk = r*np.cos(u)
        ypk = r*np.sin(u)
        xpkDot = rDot*np.cos(u) - ypk*uDot
        ypkDot = rDot*np.sin(u) + xpk*uDot

        # transform from orbital coordinate system to ECEF
        x = np.cos(u) * r * np.cos(Omega) - np.sin(u)*r * np.cos(i) * np.sin(Omega)
        y = np.cos(u) * r * np.sin(Omega) + np.sin(u)*r * np.cos(i) * np.cos(Omega)
        z = np.sin(u) * r * np.sin(i)

        # define sat positions in matrix
        sat_positions[0,sat_no] = x
        sat_positions[1,sat_no] = y
        sat_positions[2,sat_no] = z

        # compute satellite velocities
        sat_velocity[0,sat_no] = (xpkDot-ypk*np.cos(i)*OmegaDot)*np.cos(Omega) - (xpk*OmegaDot+ypkDot*np.cos(i)-ypk*np.sin(i)*iDot) * np.sin(Omega)
        sat_velocity[1,sat_no] = (xpkDot-ypk*np.cos(i)*OmegaDot)*np.sin(Omega) + (xpk*OmegaDot+ypkDot*np.cos(i)-ypk*np.sin(i)*iDot) * np.cos(Omega)
        sat_velocity[2,sat_no] = ypkDot*np.sin(i) + ypk*np.cos(i)*iDot
        
        # compute relativistic correction in clock correction
        sat_clock_relativistic_correction = nav.sel(sv=sv).SVclockDriftRate.values[idx] * dt + nav.sel(sv=sv).SVclockDrift.values[idx] * dt + nav.sel(sv=sv).SVclockBias.values[idx] + dtr

    return sat_positions, sat_velocity, sat_clock_relativistic_correction

def R1(alpha,deg=False):
    if deg:
        alpha = np.deg2rad(alpha)
    r1 = np.array([[1, 0, 0],
                   [0, np.cos(alpha), np.sin(alpha)],
                   [0, -np.sin(alpha), np.cos(alpha)]])
    return r1

def R2(alpha,deg=False):
    if deg:
        alpha = np.deg2rad(alpha)
    r2 = np.array([[np.cos(alpha), 0, -np.sin(alpha)],
                   [0, 1, 0],
                   [np.sin(alpha), 0, np.cos(alpha)]])
    return r2

def R3(alpha,deg=False):
    if deg:
        alpha = np.deg2rad(alpha)
    r3 = np.array([[np.cos(alpha), np.sin(alpha), 0],
                   [-np.sin(alpha), np.cos(alpha), 0],
                   [0, 0, 1]])
    return r3

def newMod(a, b):
    """
    Calculate the modulo of two numbers, accounting for negative dividends, unlike normal Python behavior.

    Source: https://stackoverflow.com/questions/3883004/how-does-the-modulo-operator-work-on-negative-numbers-in-python

    Input:
        a : The dividend
        b : The divisor

    Output:
        res : The modulo result
    """
    
    res = a % b
    if not np.shape(a) == ():
        idx = np.where(a<0)[0]
        res[idx] = res[idx] - b
    elif a < 0:
        res - b
    return res

def angle_2D(L1_x, L1_y, L2_x, L2_y):
    """
    Calculate the angle between two lines defined by their coordinates.

    Input:
        L1_x : X-coordinates of the first line.
        L1_y : Y-coordinates of the first line.
        L2_x : X-coordinates of the second line.
        L2_y : Y-coordinates of the second line.

    Output:
        angle : Angle between the two lines in radians.
    """
    # calculate the direction vectors of the lines
    L1_vector = np.array([L1_x[1] - L1_x[0], L1_y[1] - L1_y[0]])
    L2_vector = np.array([L2_x[1] - L2_x[0], L2_y[1] - L2_y[0]])

    # calculate the dot product of the direction vectors
    dot_product = np.dot(L1_vector, L2_vector)

    # calculate the magnitudes of the direction vectors
    L1_magnitude = np.linalg.norm(L1_vector)
    L2_magnitude = np.linalg.norm(L2_vector)

    # calculate the cosine of the angle between the lines
    cos_angle = dot_product / (L1_magnitude * L2_magnitude)

    # calculate the angle in radians
    angle = np.arccos(cos_angle)

    return angle

def angle_3D(L1_x, L1_y, L1_z, L2_x, L2_y, L2_z):
    """
    Calculate the angle between two lines defined by their coordinates.

    Input:
        L1_x : X-coordinates of the first line.
        L1_y : Y-coordinates of the first line.
        L1_z : Z-coordinates of the first line.
        L2_x : X-coordinates of the second line.
        L2_y : Y-coordinates of the second line.
        L2_z : Z-coordinates of the second line.

    Output:
        angle : Angle between the two lines in radians.
    """
    # calculate the direction vectors of the lines
    L1_vector = np.array([L1_x[1] - L1_x[0], L1_y[1] - L1_y[0], L1_z[1] - L1_z[0]])
    L2_vector = np.array([L2_x[1] - L2_x[0], L2_y[1] - L2_y[0], L2_z[1] - L2_z[0]])

    # calculate the dot product of the direction vectors
    dot_product = np.dot(L1_vector, L2_vector)

    # calculate the magnitudes of the direction vectors
    L1_magnitude = np.linalg.norm(L1_vector)
    L2_magnitude = np.linalg.norm(L2_vector)

    # calculate the cosine of the angle between the lines
    cos_angle = dot_product / (L1_magnitude * L2_magnitude)

    # calculate the angle in radians
    angle = np.arccos(cos_angle)

    return angle

def find_Fresnel_r(lambda_,rx,x,y):
    """
    Calculate the radius for the first Fresnel zone.

    Input:
        lambda_ : Wavelength in m
        rx : receiver position
        x : x-coordinate for cross-section
        y : y-coordinate for cross-section

    Output: 
        r : the approximated radius for the first Fresnel zone.
    """
    d_t = np.linalg.norm(rx - np.array([x,y]))
    r = np.sqrt(lambda_*d_t)
    return r

def loop_datetime_to_decyear(t):
    """
    function to loop through np.datetime64 vector and calculate decimal year

    input:
        t : vector with length greater than 0
    output:
        decimal_years : vector with decimal years 
    """
    # initialize vector to the same size as input
    decimal_years = np.zeros_like(t,dtype=np.float64)
    # loop through to calculate decimal years
    for i in range(len(t)):
        decimal_years[i] = datetime_to_decyear(t[i])
    return decimal_years

def datetime_to_decyear(t):
    """
    function to calculate decimal year

    input:
        t : vector with any length of type np.datetime64
    output:
        decimal_years : vector with decimal years 
    """
    # see how long the time input is
    try:
        N = len(t)
    except TypeError:
        N = 0

    # check if input is correct type
    if N == 0:
        if not isinstance(t, np.datetime64):
            raise TypeError("Input must be a numpy.datetime64 object")

    # if longer than 0, loop through the times
    if N > 0:
        decimal_year = loop_datetime_to_decyear(t)
    else:
        # get year from data
        year = np.datetime64(t,'Y').astype(int)+1970

        # convert datetime to millisecond of the calculated year
        time_of_year = np.datetime64(t,'ms') - np.datetime64(f'{year}-01-01','ms')

        # calculate the amount of milliseconds in the given year
        days_in_year = np.datetime64(f'{year + 1}-01-01','ms') - np.datetime64(f'{year}-01-01','ms')

        # calculate the decimal year
        decimal_year = year + time_of_year / days_in_year
    return decimal_year

def round_to_nearest_second(time, dt=np.timedelta64(1, 's')):
    """
    function to round to nearest second in ns precision

    input:
        time : np.datetime64 to be rounded to nearest second
    output:
        rounded time
    """
    if np.abs(np.datetime64(time,'s') - np.datetime64(time - np.datetime64('1970-01-01T00:00:00').astype(int),'ns')).astype(np.int64) > 500000000:
        return time - (time - np.datetime64('1970-01-01T00:00:00')) % dt + dt
    return time - (time - np.datetime64('1970-01-01T00:00:00')) % dt

def decimal_year_to_datetime(decimal_year):
    """
    function to calculate numpy datetime year from decimal year

    input:
        t : vector with any length of decimal year
    output:
        datetime_value : vector with numpy datetime year 
    """
    # see how long the time input is
    try:
        N = len(decimal_year)
    except TypeError:
        N = 0

    if N > 0:
        datetime_value = np.zeros_like(decimal_year)
        datetime_value = np.array(datetime_value, dtype='datetime64[ns]')
        for i in range(N):
            datetime_value[i] = decimal_year_to_datetime(decimal_year[i])
    else:
        # extract the year component from the decimal year
        year = int(decimal_year)

        # calculate the fractional part of the year
        fractional_year = decimal_year - year

        # calculate the total number of milliseconds in the year
        days_in_year = np.datetime64(f'{year + 1}-01-01','ns') - np.datetime64(f'{year}-01-01','ns')

        # calculate the milliseconds corresponding to the fractional part of the year
        ns_in_fractional_year = days_in_year * fractional_year

        # calculate the datetime for the start of the year
        start_of_year = np.datetime64(f'{year}-01-01', 'ns')

        # calculate the datetime by adding the milliseconds to the start of the year
        datetime_value = start_of_year + ns_in_fractional_year

        # round to nearest second, and give output in ns 
        datetime_value = round_to_nearest_second(datetime_value)

    return datetime_value


def remove_padding_from_Am(A_m, pad_idx, even=True):
    """
    function to remove padding from the residual multipath signal

    input:
        A_m : the residual multipath signal
        pad_idx : the indexes for the padding
        even : set if original length for signal is even
    output:
        A_m : unpadded A_m
    """
    if even:
        return A_m[pad_idx:3*pad_idx]
    else:
        return A_m[pad_idx-1:3*pad_idx]
    
def wavelet_power_for_band(coef, lower_bound, upper_bound, scales, vars):
    """
    function to calculate wavelet power for a specific band of scales

    input:
        coef : coefficients given from the wavelet function
        lower_bound : the minimum scale to be found
        upper_bound : the maximum scale to be found
        scales : the array of scales to be used
        vars : the variables to scale the averaged wavelet power with (dj, dt, Cs)
    output:
        A_m : the scale-averaged amplitude of the multipath signal
    """

    dj, dt, Cs = vars

    idx = np.where((scales >= lower_bound) & (scales < upper_bound))

    A_m = 0

    for s in idx[0]:
        A_m += np.abs(coef[s,:])**2/scales[s]
    A_m *= dj*dt/Cs

    return A_m

def pad_signal(signal,pad_length=None):
    """
    function to pad signal with a set lenght - default is half of the given signal in each end

    input:
        signal : the signal to be padded
        pad_length : the length to pad the signal with, default is half of the given signal
    output:
        signal_padded : the padded signal
    """
    if pad_length == None:
        pad_length = len(signal)//2
    
    if len(signal) % 2 == 0:
        first_pad = signal[:pad_length]
        last_pad = signal[pad_length:]
    else:
        first_pad = signal[:pad_length+1]
        last_pad = signal[pad_length:]

    signal_padded = np.concatenate([first_pad, signal, last_pad])

    return signal_padded

def calculate_multipath_error(pseudorange_target, frequency_target, carrier_phase_target, frequency_reference, carrier_phase_reference):
    """
    function to calculate the multipath error for a given pseudorange.

    input:
        pseudorange_target : the uncorrected pseudoranges for targeted signal
        frequency_target : the center frequency for the targeted signal
        carrier_phase_target : the carrier phase measurements for the targeted signal in cycles
        frequency_reference : the center frequency for the reference signal
        carrier_phase_reference : the carrier phase measurements for the reference signal in cycles

    output:
        multipath_error : the multipath error in metres
    """
    c = 299792458 # m/s
    lambda_target = c / frequency_target
    lambda_reference = c / frequency_reference

    multipath_error = pseudorange_target.copy()
    multipath_error -= ((frequency_target**2+frequency_reference**2)/(frequency_target**2-frequency_reference**2)) * (carrier_phase_target * lambda_target)
    multipath_error += ((2*frequency_reference**2)/(frequency_target**2-frequency_reference**2)) * (carrier_phase_reference * lambda_reference)
    multipath_error -= np.nanmean(multipath_error)

    return multipath_error


def least_squares_pseudoranges(pseudoranges, sat_positions, iter_max=10):
    """
    function to perform least squares on pseudoranges in order to calculate a position for a receiver

    input:
        pseudoranges : pseudoranges between the satellites and the receiver
        sat_positions : the position of the satellites
    output:
        position : the calculated position for the receiver
        receiver_clock_error : GPS receiver clock error in seconds
    """
    x = np.zeros(4,dtype=np.float64)

    for _ in range(iter_max):

        #dist = np.linalg.norm(sat_positions - x[:3])
        dist = np.sqrt((sat_positions[:,0] - x[0])**2 + (sat_positions[:,1] - x[1])**2 + (sat_positions[:,2] - x[2])**2)
        z = pseudoranges - dist - x[3]

        A = np.column_stack([-(sat_positions[:, i] - x[i]) / dist for i in range(3)] + [np.ones(len(pseudoranges))])
        h_k = np.linalg.lstsq(A,z,rcond=None)[0]
        if (np.abs(h_k) < 1e-10).all():
            break
        x += h_k

    position = x[:3]
    receiver_clock_error = x[3]
    
    return position, receiver_clock_error

def earth_rotation_correction(traveltime, satellite_position):
    """
    function to calculated a satellite postion which is corrected by the time it took to send a signal from its location to Earth.

    input:
        traveltime : the signal travel time
        satellite_position : the position of the satellite
    output:
        corrected_satellite_position : the corrected satellite position
    """

    Omegae_dot = 7.2921151467e-5 # rad/sec

    rotation_angle = Omegae_dot * traveltime

    rotation_matrix = R3(rotation_angle)

    if not np.shape(satellite_position) == (3,):
        satellite_position = satellite_position[0]

    corrected_satellite_position = rotation_matrix @ satellite_position

    return corrected_satellite_position

def ionospheric_free_combination_correction(pseudorange_target, frequency_target, pseudorange_reference, frequency_reference):
    """
    function to remove the ionospheric delay from a target pseudorange using a reference pseudorange
    method is the ionospheric free combination

    input:
        pseudorange_target : the pseudorange in which the ionospheric delay should be removed
        frequency_target : the frequency for the targeted pseudorange
        pseudorange_reference : the reference pseudorange
        frequency_reference : the frequency for the reference pseudorage
    output:
        ionospheric_corrected_pseudorange : the corrected target pseudorange
    """
    ionospheric_corrected_pseudorange = frequency_target**2 / (frequency_target**2-frequency_reference**2) * pseudorange_target 
    ionospheric_corrected_pseudorange -= frequency_reference**2 / (frequency_target**2-frequency_reference**2) * pseudorange_reference
    return ionospheric_corrected_pseudorange

def klobuchar_correction(latitude, longitude, elevation, azimuth, alpha, beta, time_epoch):
    """
    function to find the ionospheric delay from a satellite based on given ionospheric correction values in the RINEX nav file
    method is Klobuchar

    input:
        latitude : the user latitude in degrees
        longitude : the user longitude in degrees
        elevation : the satellite elevation in degrees
        azimuth : the satellite azimuth in degrees
        alpha : the alpha ionospheric corrections
        beta : the beta ionospheric corrections
        time_epoch : the current time epoch for the pseudorange in seconds
    output:
        dIon : the correction to the pseudorange in meters
    """
    c = 2.99792458e8  # speed of light
    deg2semi = 1. / 180.  # degrees to semisircles
    semi2rad = np.pi  # semisircles to radians
    
    a = np.deg2rad(azimuth)  # azimuth in radians
    e = elevation * deg2semi  # elevation angle in semicircles

    psi = 0.0137 / (e + 0.11) - 0.022  # Earth Centered angle

    lat_i = latitude * deg2semi + psi * np.cos(a)  # Subionospheric lat
    lat_i = np.clip(lat_i, -0.416, 0.416)

    long_i = longitude * deg2semi + (psi * np.sin(a) / np.cos(lat_i * semi2rad))  # Subionospheric long

    lat_m = lat_i + 0.064 * np.cos((long_i - 1.617) * semi2rad)  # Geomagnetic latitude

    t = 4.32e4 * long_i + time_epoch
    t = np.mod(t, 86400.)  # Seconds of day
    t = np.where(t > 86400., t - 86400., t)
    t = np.where(t < 0., t + 86400., t)

    sF = 1. + 16. * (0.53 - e) ** 3  # Slant factor

    PER = beta[0] + beta[1] * lat_m + beta[2] * lat_m ** 2 + beta[3] * lat_m ** 3  # Period of model
    PER = np.where(PER < 72000., 72000., PER)

    x = 2. * np.pi * (t - 50400.) / PER  # Phase of the model

    AMP = alpha[0] + alpha[1] * lat_m + alpha[2] * lat_m ** 2 + alpha[3] * lat_m ** 3  # Amplitude of the model
    AMP = np.where(AMP < 0., 0., AMP)

    dIon1 = np.where(np.abs(x) > 1.57, sF * 5.e-9, sF * (5.e-9 + AMP * (1. - x ** 2 / 2. + x ** 4 / 24.)))
    
    return c * dIon1

def saastamoinen_correction(elevation, latitude, height, pressure, temperature, relative_humidity):
    """
    function to find the tropospheric delay from a satellite based on pressure, temperature and relative humidity
    method is Saastamoinen

    input:
        elevation : the elevation of the satellite in degrees
        latitude : the user latitude in degrees
        height : the user height in km
        pressure : local atmospheric pressure in mbar
        temperature : local temperature in K
        relative_humidity : local relative humidity
    output:
        d_trop : the tropospheric delay in meters
    """
    e_s = 6.106 * relative_humidity * np.exp((17.15 * temperature - 4684) / (temperature - 38.45))  # converting RH to e_s in hPa

    latitude = np.deg2rad(latitude)  # Convert latitude to radians

    D = 1 + 0.0026 * np.cos(2 * latitude) + 0.00028 * height

    d_trop = 0.002277 * D * (pressure + (1255 / temperature + 0.05) * e_s)

    d_trop /= np.sin(np.deg2rad(elevation))
    
    return d_trop

def get_azimuth_elevation(obs, nav, galileo_nav=False, galileo_system=None, num_of_points=20):
    """
    function to return azimuth and elevation and the sv_list they are computed from

    input:
        obs: the observation data array (from georinex loader)
        nav: the navigation data array (from georinex loader)
        galileo_nav (optional): boolean to determine if the system is GPS or Galileo (default is GPS)
        galileo_system (optional): (E5A or E5B): determine which frequency to calculate for Galileo (default is None)
        num_of_points: determine how many points of each is needed for computations (default is 20)
    output:
        az: matrix of azimuth
        el: matrix of elevation
        sv_list: the satellites the above has been calculated for
    """
    sv_list = obs.sv.values

    skip_sv = []
    n = 0
    while True:
        if not n == 1:
            try:
                sv_list = [sv for sv in sv_list if sv not in skip_sv]
                n = len(sv_list)
            except TypeError:
                sv_list = str(sv_list)
                n = 1
        try:
            ecef_pos = np.array(obs.position)
            lla_pos = ecef_to_lla(ecef_pos[0],ecef_pos[1],ecef_pos[2])

            duration = (obs.time.values[-1] - obs.time.values[0]).astype('timedelta64[s]').astype(np.int64)

            t = np.linspace(0,duration,num_of_points) + (np.int64(np.nanmin(nav.TransTime.values)))

            # initialize 
            az = np.zeros((n,len(t)))
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

def generate_datetime_ns(input_string):

    day = int(input_string[:2])
    month = int(input_string[2:4])
    year = int(input_string[4:6])
    hour = int(input_string[7:9])
    minute = int(input_string[9:11])
    second = int(input_string[11:13])

    datetime_string = f'{year+2000:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}.000000000'

    datetime_ns = np.datetime64(datetime_string, 'ns')

    return datetime_ns