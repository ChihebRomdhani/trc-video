import os
import pandas as pd

output_csv = 'data_vid.csv'  
video_dir = 'downloaded_videos' 
data_csv = 'processed_data_augmented2.csv' 


selected_columns = ["pulpe de l´index (main droite)  ","pulpe de l´index (main gauche) ","l´éminence thénar  à droite ","l´éminence thénar  à gauche ","CRT1 (sec)"]

data = pd.read_csv(data_csv)

filtered_data = data[selected_columns].copy()



filtered_data.to_csv(output_csv, index=False)

print(f"Filtered data saved to {output_csv}")