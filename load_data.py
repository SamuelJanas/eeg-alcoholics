import os
import pandas as pd
import hashlib

from tqdm import tqdm

def get_dataset(data_path: str) -> pd.DataFrame:
    os.makedirs("tmp/", exist_ok=True)
    encoded_path = hashlib.sha256(data_path.encode("utf-8")).hexdigest()[:10]
    cache_path = f"tmp/{encoded_path}.pickle"

    if os.path.exists(cache_path):
        print(f"dataset loaded from {cache_path}")
        return pd.read_pickle(cache_path)

    filenames_list = os.listdir(data_path) 
    temp_data = []
    EEG_data = pd.DataFrame({}) 

    for file_name in tqdm(filenames_list):
        if not file_name.endswith(".csv"):
            continue
        temp_df = pd.read_csv(data_path + file_name, index_col=0)
        temp_data.append(temp_df)

    EEG_data = pd.concat(temp_data)

    # This will thro a warning but don't mind
    EEG_data.loc[EEG_data['matching condition'] == 'S2 nomatch,', 'matching condition'] =  'S2 nomatch'

    EEG_data.to_pickle(cache_path)
    print(f"dataset saved to {cache_path}")

    return EEG_data

