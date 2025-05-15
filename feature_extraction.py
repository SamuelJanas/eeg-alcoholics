import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from antropy import sample_entropy, hjorth_params

from load_data import get_dataset
from explore import butter_bandpass_filter, apply_bandpass_filter

def extract_all_features_per_trial(df: pd.DataFrame, subject_type: str, signal_column_name: str):
    all_features = []

    grouped = df.groupby(['name', 'trial number', 'sensor position'])

    for (name, trial, sensor), group in grouped:
        signal = group[signal_column_name].values

        # Combine all feature sets
        features = {
            **extract_time_domain_features(signal),
            **extract_frequency_domain_features(signal, fs=256),
            **extract_entropy_features(signal),
            **extract_hjorth_features(signal)
        }

        # Add metadata
        features.update({
            'subject': name,
            'trial': trial,
            'sensor': sensor,
            'group': subject_type  # 'a' for alcoholic, 'c' for control
        })

        all_features.append(features)

    return pd.DataFrame(all_features)

# TIME-DOMAIN STATISTICAL FEATURES
def extract_time_domain_features(signal):
    return {
        'mean': np.mean(signal),              # average amplitude 
        'std': np.std(signal),                # standard deviation
        'skewness': skew(signal),             # skewness: asymmetry of the signal distribution
        'kurtosis': kurtosis(signal),         # kurtosis: "peakedness" or tail-heaviness of the signal
        'ptp_amp': np.ptp(signal),            # peak-to-peak amplitude: max - min
        'rms': np.sqrt(np.mean(signal**2)),   # root mean square: signal power estimate
        'zcr': ((signal[:-1] * signal[1:]) < 0).sum()  # zero crossing rate: number of times the signal crosses zero
    }

# FREQUENCY-DOMAIN FEATURES
def extract_frequency_domain_features(signal, fs=256):
    freqs, psd = welch(signal, fs=fs, nperseg=fs)

    def bandpower(fmin, fmax):
        mask = (freqs >= fmin) & (freqs <= fmax)
        return np.sum(psd[mask])

    # divides frequency range into bands of human brain waves
    delta = bandpower(1, 4)
    theta = bandpower(4, 8)
    alpha = bandpower(8, 12)
    beta = bandpower(12, 30)
    
    total_power = np.sum(psd)

    return {
        'delta_power': delta,
        'theta_power': theta,
        'alpha_power': alpha,
        'beta_power': beta,
        'alpha_beta_ratio': alpha / beta if beta != 0 else 0,
        'theta_alpha_ratio': theta / alpha if alpha != 0 else 0,
        'spectral_entropy': -np.sum((psd/np.sum(psd)) * np.log2(psd/np.sum(psd) + 1e-8))
    }

# HJORTH PARAMETER FEATURES
def extract_hjorth_features(signal):
    mobility, complexity = hjorth_params(signal)
    return {
        'hjorth_mobility': mobility,
        'hjorth_complexity': complexity
    }

# ENTROPY FEATURES
def extract_entropy_features(signal):
    return {
        'sample_entropy': sample_entropy(signal)
    }

def main():
    train_data = get_dataset(data_path="data/SMNI_CMI_TRAIN/Train/")
    test_data = get_dataset(data_path="data/SMNI_CMI_TEST/Test/")

    combined_df = pd.concat([train_data,test_data], ignore_index=True)

    # a for alcoholics, c for control
    EEG_data_alcoholic = combined_df[combined_df['subject identifier'] == 'a']
    EEG_data_control = combined_df[combined_df['subject identifier'] == 'c']

    EEG_data_alcoholic_filtered = EEG_data_alcoholic.groupby(['name', 'trial number', 'sensor position']) \
    .apply(apply_bandpass_filter) \
    .reset_index(drop=True)

    EEG_data_control_filtered = EEG_data_control.groupby(['name', 'trial number', 'sensor position']) \
    .apply(apply_bandpass_filter) \
    .reset_index(drop=True)

    EEG_data_alcoholic_filtered_features = extract_all_features_per_trial(EEG_data_alcoholic_filtered, subject_type='a', signal_column_name='filtered_sensor_value')
    EEG_data_control_filtered_features = extract_all_features_per_trial(EEG_data_control_filtered, subject_type='c', signal_column_name='filtered_sensor_value')

    EEG_data_alcoholic_filtered_features.to_csv('data/alcoholic_features.csv', index=False)
    EEG_data_control_filtered_features.to_csv('data/control_features.csv', index=False)

if __name__ == "__main__":
    main()
