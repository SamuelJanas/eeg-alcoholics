import os
import pickle
import pandas as pd
import hashlib

from tqdm import tqdm

def get_dataset(data_path: str) -> pd.DataFrame:
    encoded_path = hashlib.sha256(data_path.encode("utf-8")).hexdigest()[:10]
    cache_path = f"tmp/{encoded_path}.pickle"

    if os.path.exists(cache_path):
        print(f"dataset loaded from {cache_path}")
        return pd.read_pickle(cache_path)

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

    EEG_data.to_pickle(cache_path)
    print(f"dataset saved to {cache_path}")

    return EEG_data

