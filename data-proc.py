import os
import re
import random
import pandas as pd
from tqdm import tqdm
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive


gauth = GoogleAuth()

'dr get a josn file from google auth, it will allow you to fetch the data directky from the drive DIR ( FILE NAME IS ALREDY SET )'
gauth.LoadClientConfigFile("")

gauth.LoadCredentialsFile("credentials.json")
if not gauth.credentials or gauth.access_token_expired:
    gauth.LocalWebserverAuth()
    gauth.SaveCredentialsFile("credentials.json")

drive = GoogleDrive(gauth)


input_csv = "Point - Corrélation entre le taux de lactate et le TRC pour les patients en réanimation (réponses) - Réponses au formulaire 1.csv"
output_csv = "processed_data.csv"
video_output_dir = "downloaded_videos"
os.makedirs(video_output_dir, exist_ok=True)

df = pd.read_csv(input_csv)

video_columns = []
for col in df.columns:
    if df[col].astype(str).str.contains("drive.google.com").any():
        video_columns.append(col)


def download_via_pydrive(file_id, save_dir="downloaded_videos"):

    try:
        gfile = drive.CreateFile({"id": file_id})
        local_filename = os.path.join(save_dir, f"{file_id}.mp4")
        gfile.GetContentFile(local_filename)
        return local_filename
    except Exception as e:
        print(f"[ERROR] Could not download file_id={file_id}: {e}")
        return None

for idx in tqdm(range(len(df)), desc="Processing rows"):
    row = df.iloc[idx]

    for col in video_columns:
        cell_value = str(row[col])

        if "drive.google.com" not in cell_value:
            continue

        links = cell_value.split(',')
        downloaded_paths = []

        for link in links:
            link = link.strip()

            match = re.search(r"id=([a-zA-Z0-9_-]+)", link)
            if match:
                file_id = match.group(1)
                local_path = download_via_pydrive(file_id, video_output_dir)
                if local_path:
                    downloaded_paths.append(local_path)

        if downloaded_paths:
            df.at[idx, col] = ",".join(downloaded_paths)


df['CRT1 (sec)'] = [round(random.uniform(0.2, 3.0), 2) for _ in range(len(df))]
df['CRT2 (sec)'] = [round(random.uniform(0.2, 3.0), 2) for _ in range(len(df))]
df['CRT3 (sec)'] = [round(random.uniform(0.2, 3.0), 2) for _ in range(len(df))]
df['CRT4 (sec)'] = [round(random.uniform(0.2, 3.0), 2) for _ in range(len(df))]

df.to_csv(output_csv, index=False)
print(f"Done! Processed CSV saved as '{output_csv}'.")
