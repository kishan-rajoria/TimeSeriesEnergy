
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np

def detect_anomalies(df, columns):
    anomaly_data = {
        'column_name': [],
        'negative_values': [],
        'spikes': [],
        'zeros': [],
        'nan_values': []
    }
    
    for col in columns:
        if col in df.columns:
            col_data = df[col]
            negative_values_count = (col_data < 0).sum()
            mean = col_data.mean()
            std = col_data.std()
            spike_threshold = 4 * std
            spikes_count = (np.abs(col_data - mean) > spike_threshold).sum()
            zeros_count = (col_data == 0).sum()
            nan_values_count = col_data.isna().sum()
            anomaly_data['column_name'].append(col)
            anomaly_data['negative_values'].append(negative_values_count)
            anomaly_data['spikes'].append(spikes_count)
            anomaly_data['zeros'].append(zeros_count)
            anomaly_data['nan_values'].append(nan_values_count)
    anomaly_df = pd.DataFrame(anomaly_data)
    
    return anomaly_df

def process_column_names(columns):
    processed_columns = []
    for col in columns:
        col = col.replace("WEATHER STATION.", "")
        col = col.replace(".", "_")
        col = col.replace("DEGC", "c")
        col = col.replace("AMBIENT", "amb")
        col = col.replace("IRRADIANCE", "")
        col = col.replace("RELATIVE_HUMIDITY", "rh")
        col = col.replace("WIND_DIRECTION", "wind_dir")
        col = col.replace("WIND_SPEED", "wind_speed")
        col = col.lower()
        processed_columns.append(col)
    return processed_columns

def plot_features(df, x_feature, y_features, duration):
    if isinstance(duration, str):
        start, end = map(int, duration.split(":"))
    else:
        start, end = 0, duration
    start = max(0, start)
    end = min(len(df), end)
    data_subset = df.iloc[start:end]
    fig = go.Figure()
    for y_feature in y_features:
        if y_feature in df.columns:
            fig.add_trace(go.Scatter(x=data_subset[x_feature], 
                                        y=data_subset[y_feature], 
                                        mode='lines', 
                                        name=y_feature))
    fig.update_layout(
        title=f'Plot of {y_features} over {x_feature} (Range: {start}:{end})',
        xaxis_title=x_feature,
        yaxis_title='Values',
        legend_title="Features",
        template="plotly_dark",
        hovermode='x unified'
    )
    fig.show()
    
def convert_kw_to_mw(df, column_name):
    if column_name in df.columns:
        df[column_name] = df[column_name] / 1000
    else:
        print(f"Column '{column_name}' not found in the DataFrame")
    return df

def impute_negative_values(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: 0 if x < 0 else x)
    return df

def impute_spikes(df, columns, spike_threshold=3):
    for col in columns:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            spike_condition = (df[col] > mean + spike_threshold * std) | (df[col] < mean - spike_threshold * std)
            df.loc[spike_condition, col] = mean
    return df

def fill_nan_with_zero(df):
    df.fillna(0, inplace=True)
    return df

def second_stage_imputation(df, columns, spike_threshold=4):
    spike_columns = identify_spikes(df, columns, spike_threshold)
    while spike_columns:
        print(f"Imputing spikes in columns: {spike_columns}")
        df = impute_spikes(df, spike_columns, spike_threshold)
        spike_columns = identify_spikes(df, spike_columns, spike_threshold)
    return df

def identify_spikes(df, columns, spike_threshold=4):
    spike_columns = []
    
    for col in columns:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            spike_condition = (df[col] > mean + spike_threshold * std) | (df[col] < mean - spike_threshold * std)
            if df[spike_condition].shape[0] > 0:
                spike_columns.append(col)
    
    return spike_columns

def impute_zeros_with_sklearn(df, cols, strategy='mean'):
    """
    Impute zero values in specified columns using sklearn's SimpleImputer.
    Available strategies: 'mean', 'median', 'most_frequent', 'constant'
    
    Parameters:
    - df: DataFrame containing the data.
    - cols: List of column names to be imputed.
    - strategy: Imputation strategy for SimpleImputer.
    
    Returns:
    - DataFrame with imputed values.
    """
    if not isinstance(cols, list):
        raise TypeError("Column names must be provided as a list of strings.")
    missing_cols = [col for col in cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Columns not found in DataFrame: {', '.join(missing_cols)}")
    imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
    
    for col in cols:
        df[col] = df[col].replace(0, np.nan)
        df[[col]] = imputer.fit_transform(df[[col]])
    return df

def impute_radiation_based_on_solar_power(df, solar_power_col, radiation_cols):
    """
    Impute radiation values based on the solar power values.
    
    Parameters:
    - df: DataFrame containing the data.
    - solar_power_col: Name of the solar power column.
    - radiation_cols: List of column names for solar radiation.
    
    Returns:
    - DataFrame with imputed radiation values.
    """
    if not isinstance(radiation_cols, list):
        raise TypeError("Radiation columns must be provided as a list of strings.")
    
    if solar_power_col not in df.columns:
        raise KeyError(f"Solar power column '{solar_power_col}' not found in DataFrame.")
    
    for col in radiation_cols:
        if col not in df.columns:
            raise KeyError(f"Radiation column '{col}' not found in DataFrame.")
    for i in range(len(df)):
        if df.loc[i, solar_power_col] == 0:
            df.loc[i, radiation_cols] = 0
        else:
            for col in radiation_cols:
                if df.loc[i, col] == 0:
                    df.loc[i, col] = np.nan  
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    df[radiation_cols] = imputer.fit_transform(df[radiation_cols])
    
    return df

def count_negative_radiation_during_negative_solar(df, time_col, solar_kw_col, radiation_cols):
    """
    Count the occurrences of negative values in radiation columns during periods when the solar power is negative.
    
    Parameters:
    - df: DataFrame containing the data.
    - time_col: Name of the time column.
    - solar_kw_col: Name of the solar power column.
    - radiation_cols: List of column names for radiation.
    
    Returns:
    - DataFrame with counts of negative values in radiation columns during negative solar power periods.
    """
    if not isinstance(radiation_cols, list):
        raise TypeError("Radiation columns must be provided as a list of strings.")
    
    if solar_kw_col not in df.columns:
        raise KeyError(f"Solar power column '{solar_kw_col}' not found in DataFrame.")
    
    if time_col not in df.columns:
        raise KeyError(f"Time column '{time_col}' not found in DataFrame.")
    
    for col in radiation_cols:
        if col not in df.columns:
            raise KeyError(f"Radiation column '{col}' not found in DataFrame.")
    negative_solar_df = df[df[solar_kw_col] < 0]
    negative_solar_df = negative_solar_df.sort_values(by=time_col)
    negative_counts = {col: 0 for col in radiation_cols}
    for _, row in negative_solar_df.iterrows():
        for col in radiation_cols:
            if row[col] < 0:
                negative_counts[col] += 1
    counts_df = pd.DataFrame(list(negative_counts.items()), columns=['Radiation Column', 'Negative Count'])
    return counts_df

def impute_radiation_based_on_solar_power_2(df, solar_power_col, radiation_cols):
    """
    Impute radiation values based on the solar power values.
    
    Parameters:
    - df: DataFrame containing the data.
    - solar_power_col: Name of the solar power column.
    - radiation_cols: List of column names for radiation.
    
    Returns:
    - DataFrame with imputed radiation values.
    """
    if not isinstance(radiation_cols, list):
        raise TypeError("Radiation columns must be provided as a list of strings.")
    
    if solar_power_col not in df.columns:
        raise KeyError(f"Solar power column '{solar_power_col}' not found in DataFrame.")
    
    for col in radiation_cols:
        if col not in df.columns:
            raise KeyError(f"Radiation column '{col}' not found in DataFrame.")
    for col in radiation_cols:
        df.loc[df[solar_power_col] == 0, col] = 0
    for col in radiation_cols:
        df.loc[df[solar_power_col] != 0, col] = df[df[solar_power_col] != 0][col].replace(0, np.nan)
    for col in radiation_cols:
        non_zero_values = df[df[col] != 0][col]
        mean_value = non_zero_values.mean() if not non_zero_values.empty else np.nan
        if not np.isnan(mean_value):
            df[col].fillna(mean_value, inplace=True)
    return df

def copy_average_to_colA(df, colA, cols_list):
    """
    This function takes a DataFrame, calculates the row-wise average of the given list of columns,
    and assigns that average to the specified column (colA).
    
    Parameters:
    - df: DataFrame containing the data.
    - colA: The target column where the average will be copied.
    - cols_list: List of columns from which the average will be calculated.
    
    Returns:
    - The DataFrame with updated colA values.
    """
    if colA not in df.columns:
        raise KeyError(f"Target column '{colA}' not found in DataFrame.")
    for col in cols_list:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' in cols_list not found in DataFrame.")
    df[colA] = df[cols_list].mean(axis=1)
    
    return df
