import pandas as pd

from load_data import get_dataset
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt

FS = 256

def butter_bandpass_filter(
    data: pd.Series, 
    lowcut: int, 
    highcut: int, 
    fs: int, 
    order: int=5,
):
    #nyq = 0.5 * fs: The Nyquist frequency - represents the highest frequency that can be analyzed.
    nyq = 0.5 * fs   
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)

    return y

def apply_bandpass_filter(df: pd.DataFrame) -> pd.DataFrame:
    #lowcut and highcut: Lower and upper cutoff frequencies for the bandpass filter (e.g., 1 Hz to 30 Hz).
    df['filtered_sensor_value'] = butter_bandpass_filter(df['sensor value'], 1, 30, FS)
    return df

def plot_filtered_data(df_alcohol, df_control, sensors):
    for sensor in sensors:
        plt.figure(figsize=(15, 5))
        for condition in df_alcohol['matching condition'].unique():
            plt.subplot(
                1, 
                len(df_alcohol['matching condition'].unique()),
                list(df_alcohol['matching condition'].unique()).index(condition) + 1,
            )

            # Plot for alcohol group
            subset_alcohol = df_alcohol[
                (df_alcohol['sensor position'] == sensor) & 
                (df_alcohol['matching condition'] == condition)
            ]

            if not subset_alcohol.empty:
                subset_alcohol.groupby('time')['filtered_sensor_value'].mean() \
                .plot( label='Alcohol Group', color='blue', linewidth=1.5)

            # Plot for control group
            subset_control = df_control[
                (df_control['sensor position'] == sensor) &
                (df_control['matching condition'] == condition)
            ]

            if not subset_control.empty:
                subset_control.groupby('time')['filtered_sensor_value'].mean() \
                .plot( label='Control Group', color='orange', linewidth=1.5)

            plt.title(f'Sensor {sensor} - Condition: {condition}')
            plt.xlabel('Time (s)')
            plt.ylabel('Filtered Sensor Value (µV)')

            if not subset_alcohol.empty or not subset_control.empty:
                plt.legend()

            plt.tight_layout()
            plt.savefig(f"figs/{sensor}.png")

def main():
    train_data = get_dataset(data_path="data/SMNI_CMI_TRAIN/Train/")
    test_data = get_dataset(data_path="data/SMNI_CMI_TEST/Test/")

    combined_df = pd.concat([train_data,test_data], ignore_index=True)

    # a for alcoholics, c for control
    EEG_data = combined_df[combined_df['subject identifier'] == 'a']
    EEG_data_control = combined_df[combined_df['subject identifier'] == 'c']

    EEG_data_filtered = EEG_data.groupby(['name', 'trial number', 'sensor position']) \
    .apply(apply_bandpass_filter) \
    .reset_index(drop=True)

    EEG_data_control_filtered = EEG_data_control.groupby(['name', 'trial number', 'sensor position']) \
    .apply(apply_bandpass_filter) \
    .reset_index(drop=True)

    sensors_to_plot = ['AF1', 'FP1', 'CZ']
    plot_filtered_data(
       df_alcohol = EEG_data_filtered, 
       df_control = EEG_data_control_filtered, 
       sensors = sensors_to_plot,
    )

if __name__ == "__main__":
    main()

