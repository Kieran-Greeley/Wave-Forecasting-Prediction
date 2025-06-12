import pandas as pd
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder

consolidated_csv_path = '/WAVE/projects/CSEN-140-Sp25/team_10/consolidated_wave_forecast.csv'

df = pd.read_csv(consolidated_csv_path)
print("Data loaded successfully.")
print(f"Initial dataframe shape: {df.shape}")

print("\n columns:")
print(df.columns.tolist())
print(df.info())

columns_to_drop = ['date',  'moon_phase', 'iddayforecast', 'idbeach', 'idtide', 'time', 'idhourforecast', 
                    'temperature', 'feelslike', 'name', 'city', 'state', 'country', 'latitude', 'longitude',
                    'idspot', 'datetime', 'name_spotdata', 'datetime_tide', 'idseaConditionFact']

time_columns_to_convert = ['sunrise', 'sunset', 'moonset', 'moonrise']
columns_to_onehot_encode = ['type']

def time_to_minutes(time_str):
    if pd.isna(time_str):
        return None
    time_str = str(time_str).strip()
    if 'no' in time_str.lower() or time_str == '':
        return None
    try:
        time_obj = datetime.strptime(time_str, '%I:%M %p')
        return time_obj.hour * 60 + time_obj.minute
    except ValueError:
        return None
    
for column in time_columns_to_convert:
    if column in df.columns:
        df[column] = df[column].apply(time_to_minutes)
        average_minutes = df[column].mean()
        df[column] = df[column].fillna(average_minutes)

encoder = OneHotEncoder(sparse_output=False, drop='first')
for column in columns_to_onehot_encode:
    encoded = encoder.fit_transform(df[[column]])
    feature_names = encoder.get_feature_names_out([column])
    encoded_df = pd.DataFrame(encoded, columns=feature_names, index=df.index)
    df = pd.concat([df, encoded_df], axis=1)
    df = df.drop(columns=[column])

df_cleaned = df.drop(columns=columns_to_drop, errors='ignore')
print(f"\nShape after dropping columns: {df_cleaned.shape}")

print("\nRemaining columns:")
print(df_cleaned.columns.tolist())
print(df_cleaned.info())

print("\nFirst 10 rows of the cleaned dataframe:")
print(df_cleaned.head(10))

print("\nSaving dataframe to prepoccesed_wave_forecast.csv")
df_cleaned.to_csv('prepoccesed_wave_forecast.csv', index=False)