import os
import pandas as pd

from tqdm import tqdm

def get_dataset(data_path: str) -> pd.DataFrame:
    filenames_list = os.listdir(data_path) 
    EEG_data = pd.DataFrame({}) 

    for file_name in tqdm(filenames_list):
        if not file_name.endswith(".csv"):
            continue
        temp_df = pd.read_csv(data_path + file_name)
        EEG_data = pd.concat([EEG_data, temp_df])

    EEG_data = EEG_data.drop(columns=["Unnamed: 0"])
    # This will thro a warning but don't mind
    EEG_data.loc[EEG_data['matching condition'] == 'S2 nomatch,', 'matching condition'] =  'S2 nomatch'
    return EEG_data

