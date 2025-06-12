import pandas as pd
import os

# --- Configuration ---
# Define the output path for the consolidated CSV
output_csv_path = 'consolidated_wave_forecast.csv'

# Base path for your dataset files
base_data_path = '/WAVE/scratch/CSEN-140-Sp25/team_10_dataset/'

# Define the names of your CSV files
file_names = ['day_forecast', 'tide', 'hour_forecast', 'beach', 'spot', 'sea_condition_fact']

# --- Load Data ---
print("Loading data from CSV files...")
dataframes = {}
for name in file_names:
    path = os.path.join(base_data_path, name + '.csv')
    if not os.path.exists(path):
        print(f"Error: File not found at {path}")
        exit()
    try:
        df = pd.read_csv(path, sep=";")
        dataframes[name] = df
        print(f"Loaded {name}. Shape: {dataframes[name].shape}")
    except Exception as e:
        print(f"Error loading {name} from {path}: {e}")
        exit()

# Assign dataframes to variables for easier access
df_day_forecast = dataframes['day_forecast']
df_tide = dataframes['tide']
df_hour_forecast = dataframes['hour_forecast']
df_beach = dataframes['beach']
df_spot = dataframes['spot']
df_sea_condition_fact = dataframes['sea_condition_fact']

# --- Perform Merges ---
print("Starting data merging process...")

# Merge tide onto day_forecast using 'iddayforecast'
print("Merging tide onto day_forecast...")
merged_df = pd.merge(df_day_forecast, df_tide, on='iddayforecast', how='left', suffixes=('_day', '_tide'))
print(f"Shape after merging tide: {merged_df.shape}")

# Merge hour_forecast onto the current merged dataframe using 'iddayforecast'
print("Merging hour_forecast onto merged dataframe...")
merged_df = pd.merge(merged_df, df_hour_forecast, on='iddayforecast', how='left', suffixes=('', '_hour'))
merged_df.columns = merged_df.columns.str.replace('__hour$', '', regex=True)
print(f"Shape after merging hour_forecast: {merged_df.shape}")

# Create datetime column in the main merged_df
# Combine 'date' from day_forecast and 'time' from hour_forecast
# Ensure 'time' is treated as string and padded for consistent format
merged_df['time_str'] = merged_df['time'].astype(str).str.zfill(4)
merged_df['datetime'] = pd.to_datetime(merged_df['date'] + ' ' + merged_df['time_str'].str.slice(0, 2) + ':' + merged_df['time_str'].str.slice(2, 4), errors='coerce')
merged_df = merged_df.drop(columns=['time_str']) # Drop the temporary time string column

# Merge beach onto the current merged dataframe using 'idbeach'
print("Merging beach onto merged dataframe...")
merged_df = pd.merge(merged_df, df_beach, on='idbeach', how='left', suffixes=('', '_beach'))
merged_df.columns = merged_df.columns.str.replace('__beach$', '', regex=True)
print(f"Shape after merging beach: {merged_df.shape}")

# Merge sea_condition_fact onto spot using 'idspot' first
print("Merging sea_condition_fact onto spot...")
df_spot_data = pd.merge(df_spot, df_sea_condition_fact, on='idspot', how='left', suffixes=('_spot', '_sea'))
print(f"Shape of merged spot_data: {df_spot_data.shape}")

# Convert 'date' column in df_spot_data to datetime
df_spot_data['date'] = pd.to_datetime(df_spot_data['date'], errors='coerce')
df_spot_data = df_spot_data.rename(columns={'date': 'datetime_spot'}) # Rename to avoid conflict

# Merge combined spot_data onto the main merged dataframe using 'idbeach' and the datetime columns
print("Merging spot_data onto main merged dataframe using 'idbeach' and datetime...")
# Drop redundant idspot column from df_spot_data before merging on idbeach
if 'idspot_spot' in df_spot_data.columns:
     df_spot_data = df_spot_data.drop(columns=['idspot_spot'], errors='ignore')
elif 'idspot' in df_spot_data.columns and 'idspot' != 'idbeach':
     df_spot_data = df_spot_data.drop(columns=['idspot'], errors='ignore')


merged_df = pd.merge(merged_df, df_spot_data, left_on=['idbeach', 'datetime'], right_on=['idbeach', 'datetime_spot'], how='left', suffixes=('', '_spotdata'))
merged_df.columns = merged_df.columns.str.replace('__spotdata$', '', regex=True)
# Drop the redundant datetime_spot column after merging
merged_df = merged_df.drop(columns=['datetime_spot'], errors='ignore')
print(f"Shape after merging spot data: {merged_df.shape}")


# --- Verification ---
print("\nMerge process complete.")
print(f"Final consolidated dataframe shape: {merged_df.shape}")
print("\nFirst 5 rows of the consolidated dataframe:")
print(merged_df.head())

print("\nChecking for missing values introduced by left merges:")
print(merged_df.isnull().sum().sort_values(ascending=False).head())

# --- Save Consolidated Data ---
print(f"\nSaving consolidated data to {output_csv_path}...")
try:
    merged_df.to_csv(output_csv_path, index=False)
    print("Consolidated CSV saved successfully.")
except Exception as e:
    print(f"Error saving consolidated CSV: {e}")
