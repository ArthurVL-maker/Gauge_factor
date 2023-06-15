# gauge_factor.py
# ----------------------------------------------------------------------------------------------------------
# Uses the recorded signals from the strain gauges of the input bar to calculate the gauge factor
# for split-Hopkinson pressure bar experiments.

# INPUTS:
# - raw_file: Path to csv file containing oscilloscope data columns Time, Ch1, Ch2, Ch3, Ch4 (string).
# - input_bar_gauge_channel: Input bar oscilloscope channel number.

# OUTPUTS:
# - gauge factor: gauge factor of the input bar.
# - wave speed: wave speed in the input bar.

# Authors: Arthur van Lerberghe (avanlerberghe1@sheffield.ac.uk) & Kin Shing Oswald Li (ksoli1@sheffield.ac.uk)
# ----------------------------------------------------------------------------------------------------------
# Imported modules:
from pathlib import Path
import numpy as np
import pandas as pd
import statistics
import warnings

# Filter warnings:
warnings.filterwarnings("ignore")


def gauge_factor(raw_file, input_bar_gauge_channel):
    # Inputs:
    length = 1000  # Length from gauge to reflection interface, mm
    velocity = 5.5  # Velocity of striker from speed trap, m/s
    voltage_input = 4  # Voltage from input power, V
    bar_amplification = 10  # Amplification factor of input bar
    time_delay = 370  # Starts detecting reflected wave only after specified timesteps from incident_start to avoid noise interference (0 if no noise after incident pulse)

    # CSV file format: Relative time, Channel 1, Channel 2, Channel 3 & Channel 4.
    raw_data = pd.read_csv(raw_file, sep=';', skiprows=9, header=None)  # Read csv file.
    time_base = raw_data.iloc[1:3, 0]  # First two time values, sec.
    in_bar_gauge_signal = raw_data[input_bar_gauge_channel].iloc[1:50000]  # Input bar signal, V
    
    # Strain gauge signals:
    time_step = time_base[2] - time_base[1]  # Oscilloscope time step, s.
    in_bar_gauge_zero = statistics.mean(in_bar_gauge_signal.iloc[:1000])  # Mean input bar "no signal" voltage, V.
    in_bar_gauge_signal_n = in_bar_gauge_signal - in_bar_gauge_zero  # Input bar signal corrected for "no signal" voltage, V
    
    # Detect pulses:
    trigger_voltage = 0.05  # Voltage to trigger detection of pulse, V
    zero_voltage = 0.01  # Voltage to trigger detection of zero pulse, V

    # Incident pulse:
    incident_trigger = np.where(abs(in_bar_gauge_signal_n) > trigger_voltage)[0][0]  # Find when the signal is first greater than trigger_voltage
    if in_bar_gauge_signal[incident_trigger] < 0:
        in_bar_gauge_signal = -in_bar_gauge_signal  # If incident wave is negative, invert signal.

    incident_start = np.where(np.diff(np.sign(in_bar_gauge_signal_n[:incident_trigger])) != 0)[-1][-1]  # Find last change of sign before trigger (start of incident pulse).
    incident_end = np.where(np.diff(np.sign(in_bar_gauge_signal_n[incident_start:])) != 0)[0][1] + incident_start  # Find the next change of sign after trigger (end of incident pulse).

    # Reflected pulse:
    if time_delay > 0:
        detect_reflected = incident_start + time_delay
    else:
        detect_reflected = incident_end         
   
    reflected_trigger = np.where(abs(in_bar_gauge_signal_n.iloc[detect_reflected:]) > trigger_voltage)[0][1] + detect_reflected - 1  # Find when signal next has a value larger than trigger_strain.
    reflected_start = np.where(abs(in_bar_gauge_signal_n.iloc[detect_reflected:reflected_trigger]) < zero_voltage)[0][-1] + detect_reflected  # Find the last "zero" before the trigger (start of reflected pulse).

    # Voltage processing:
    max_incident = np.argmax(abs(in_bar_gauge_signal_n[incident_start:incident_end])) + incident_start + 1  # Index where voltage is maximum in incident pulse
    voltage_output = in_bar_gauge_signal_n[max_incident]  # Flat voltage on top of the pulse, V
    voltage_ratio = abs(voltage_output/voltage_input)  # Ratio between output and input voltage, unitless
    
    # Gauge factor processing:
    time_difference = abs((reflected_start - incident_start)*time_step)  # Difference in time between incident and reflected waves, sec
    wavespeed_c = 2 * length / (1000 * time_difference)  # Wave speed in the bar, m/s
    strain_bar = velocity / (2 * wavespeed_c)  # Theoretical strain of bar, unitless
    gauge_factor = 2 * voltage_ratio / (strain_bar * bar_amplification)  # Gauge factor of bar, unitless

    # Print results:
    file = Path(raw_file)
    print('-' * 80 + '\n' + f"PROCESSING GAUGE FACTOR & WAVESPEED OF: {file.parts[-1].split('.')[0]} " '\n' + '-' * 80)
    print(f'Original file path:{file}')

    return f'Gauge Factor: {round(gauge_factor)} \nWave speed: {round(wavespeed_c)} m/s \n'
