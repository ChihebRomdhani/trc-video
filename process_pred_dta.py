import pandas as pd
import argparse

def clean_csv(csv_path, output_csv):
    # Load the CSV data
    data = pd.read_csv(csv_path, delimiter=',')
    print(f"Loaded {len(data)} rows from '{csv_path}'.")

    # Define the columns to drop
    drop_cols = {
        "Horodateur",
        "Matricule du patient",
        "Date du point hémodynamique",
        "Heure du point hémodynamique",
        "pulpe de l´index (main droite)",
        "pulpe de l´index (main gauche)",
        "l´éminence thénar  à droite",
        "l´éminence thénar  à gauche",
        "CRT2 (sec)",
        "CRT3 (sec)",
        "CRT4 (sec)"
    }
    
    # Drop the columns if they exist; errors='ignore' ensures that columns not present are skipped.
    data_clean = data.drop(columns=[col for col in data.columns if col.strip() in drop_cols], errors='ignore')

    # Save the cleaned DataFrame to a new CSV file.
    data_clean.to_csv(output_csv, index=False)
    print(f"Cleaned CSV saved to '{output_csv}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean CSV by dropping unnecessary columns")
    parser.add_argument("--csv_path", type=str, default="processed_data_augmented2.csv",
                        help="Path to the original CSV file")
    parser.add_argument("--output_csv", type=str, default="processed_data_cleaned.csv",
                        help="Path to save the cleaned CSV file")
    args = parser.parse_args()
    
    clean_csv(args.csv_path, args.output_csv)
