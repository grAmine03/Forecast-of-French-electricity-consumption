import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
import random
import os
import joblib
import	seaborn	as	sns
from sklearn.impute import KNNImputer
import holidays
from scipy.signal import savgol_filter

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import math
from torch.utils.checkpoint import checkpoint
from Models import *
#from Training_loops import *

data_train = pd.read_csv("train.csv")
data_test = pd.read_csv("test.csv")
meteo = pd.read_parquet("meteo.parquet")

"""# Data visualisation"""

data_train.head()
print("Data Info")
data_train.info()

"""#### NaN values"""

#plt.figure(figsize=(10,	6))
#sns.heatmap(data_train.isna(),	cbar=False,	cmap="viridis")
#plt.title("NaN values location")

nan_percentage	=	data_train.isna().mean()*100
print("\nPercentage	of	missing	values	before	processing:\n",	nan_percentage)

"""#### Comsuption of France Over Time"""

date = pd.to_datetime(data_train['date'])

#plt.figure(figsize=(12, 6))
#plt.plot(date, data_train['France'])
#plt.xlabel('Date')
#plt.ylabel('France')
#plt.title('Comsuption of France Over Time')
#plt.grid(True)
#plt.xticks(rotation=45)
#plt.tight_layout()
#plt.show()

"""# Feature engineering

### NaN values
"""
print("Feature Engineering")
# Drop 'date' column (non-numeric) for imputation
df_numeric = data_train.drop(columns=['date'])
sauv_date = data_train['date']

# Initialize KNN Imputer with 5 neighbors
imputer = KNNImputer(n_neighbors=5)

# Apply imputation
print("KNN Imputer running...")
data_train = pd.DataFrame(imputer.fit_transform(df_numeric), columns=df_numeric.columns)
data_train.insert(0, 'date',  sauv_date)
print("Imputing completed!")
data_train.isnull().sum()

"""### Features extraction and creation"""

######################################################
#### Fonction création données temporelles
######################################################

def date_data(df, date_col='date'):
    """
    Adds temporal features to a DataFrame based on a date column.

    Parameters:
    -----------
    - df (pd.DataFrame): DataFrame containing the data.
    - date_col (str): Name of the date column in the DataFrame. Default is 'date'.

    Returns:
    -------
    - df (pd.DataFrame): DataFrame with added temporal features.

    Description:
    ------------
    This function enhances a DataFrame by adding temporal features derived from a date column.
    These features include year, month, day, day of the week, and cyclical features for day of the year,
    day of the week, and hour. Additionally, it categorizes the date into seasons and identifies French holidays
    and weekends.
    The cyclical features are computed using sine and cosine transformations to capture the periodic nature
    of time.
    """

    df[date_col] = pd.to_datetime(df[date_col], utc=True)
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['day'] = df[date_col].dt.day
    df['dayofweek'] = df[date_col].dt.day_of_week
    #df['hour'] = df[date_col].dt.hour
    #df['minute'] = df[date_col].dt.minute
    df['dayofyear_sin'] = np.sin(2 * np.pi * df['date'].dt.dayofyear / 365)
    df['dayofyear_cos'] = np.cos(2 * np.pi * df['date'].dt.dayofyear / 365)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['hour_sin'] = np.sin(2 * np.pi * df[date_col].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df[date_col].dt.hour / 24)

    df['saison'] = df[date_col].dt.month.map({
        12: 1, 1: 1, 2: 1,  # Hiver (Décembre - Février)
        3: 2, 4: 2, 5: 2,          # Printemps (Mars - Mai)
        6: 3, 7: 3, 8: 3,          # Été (Juin - Août)
        9: 4, 10: 4, 11:4              # Automne (Septembre - Novembre)
    })

    # Ajout des jours fériés français
    french_holidays = holidays.FR()
    df['holiday'] = df[date_col].dt.date.apply(lambda x: 1 if x in french_holidays else 0)
    df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)

    # Pour ajouter les ponts si on veut
    """
    # Ajouter colonne "pont" : si le jeudi est férié, le vendredi suivant est un pont
    df['pont'] = 0  # Initialisation à 0
    df['previous_day'] = df[date_col] - pd.Timedelta(days=1)  # Colonne avec la veille
    df['previous_holiday'] = df['previous_day'].dt.date.apply(lambda x: 1 if x in french_holidays else 0)
    df.loc[(df['dayofweek'] == 4) & (df['previous_holiday'] == 1), 'pont'] = 1  # Vendredi après un jeudi férié

    # Suppression des colonnes temporaires
    df.drop(columns=['previous_day', 'previous_holiday'], inplace=True)

    """
    return df

######################################################
#### Fonction ajout des conso année dernière
######################################################

def add_previous_year_values(final_train, final_test, target_columns):
    """
    Adds new columns to final_train and final_test containing the consumption values
    of the same day and same hour from the previous year for each target column.

    - For each target column, the function creates a shifted version (one year back) based on the date column.
    - The 2021 values are used to fill in final_test, while previous years (2017-2020) are concatenated and merged with final_train.
    - February 29, 2020, is removed to avoid misalignment.

    Parameters:
    final_train (pd.DataFrame): Training dataset with a 'date' column and target columns.
    final_test (pd.DataFrame): Test dataset with a 'date' column (without target values).
    target_columns (list): List of target columns representing electricity consumption.

    Returns:
    tuple: Updated final_train and final_test DataFrames with new lagged columns.
    """

    final_train = final_train.copy()
    final_test = final_test.copy()

    # Remove February 29, 2020, to align data correctly
    final_train = final_train[~((final_train['date'].dt.month == 2) & (final_train['date'].dt.day == 29) & (final_train['date'].dt.year == 2020))]
    final_train = final_train.reset_index(drop=True)

    for col in target_columns:
        df_2016 = final_train[final_train['year'] == 2017][['date', col]]
        df_2017 = final_train[final_train['year'] == 2017][['date', col]]
        df_2018 = final_train[final_train['year'] == 2018][['date', col]]
        df_2018bis = final_train[(final_train['year'] == 2018) & (final_train['date'] >= '2018-01-01') & (final_train['date'] <= '2018-02-13')][['date', col]]
        df_2019 = final_train[final_train['year'] == 2019][['date', col]]
        df_2020 = final_train[final_train['year'] == 2020][['date', col]]
        df_2021 = final_train[final_train['year'] == 2021][['date', col]]

        new_rows = pd.DataFrame({'date': pd.to_datetime(['2021-12-31 23:00:00+00:00', '2021-12-31 23:30:00+00:00']), col: [54442.0, 54442.0]})
        df_2021 = pd.concat([df_2021, new_rows], ignore_index=True)
        new_rows = pd.DataFrame({'date': pd.to_datetime(['2020-12-31 23:00:00+00:00', '2020-12-31 23:30:00+00:00']), col: [64390.0, 64223.0]})
        df_2021 = pd.concat([new_rows, df_2021], ignore_index=True)

        # Shift dates by one year
        df_2017['date'] = df_2017['date'] + pd.DateOffset(years=1)
        df_2018['date'] = df_2018['date'] + pd.DateOffset(years=1)
        df_2019['date'] = df_2019['date'] + pd.DateOffset(years=1)
        df_2020['date'] = df_2020['date'] + pd.DateOffset(years=1)
        df_2021['date'] = df_2021['date'] + pd.DateOffset(years=1)

        # Combine previous years into one dataset
        df_combined = pd.concat([df_2016, df_2017, df_2018, df_2018bis, df_2019, df_2020], ignore_index=True).sort_values(by='date')
        df_combined = df_combined.iloc[:-2]  # Remove last two rows (to match dataset structure)
        df_combined = df_combined.rename(columns={col: f'{col}_prev_year'})

        # Merge back into final_train and final_test
        final_train = pd.merge(final_train, df_combined[['date', f'{col}_prev_year']], on='date', how='left')
        final_test = pd.merge(final_test, df_2021[['date', col]], on='date', how='left')
        final_test = final_test.rename(columns={col: f'{col}_prev_year'})

    return final_train, final_test


######################################################
#### Fonction ajout des données météo
######################################################

def add_meteo_feature(final_train, final_test, meteo, feature, rename_suffix, train_date, test_date, index_col='numer_sta'):
    """
    Adds a meteorological variable (e.g., wind or pressure) to the training and test datasets.

    Parameters:
    - final_train (pd.DataFrame): Training dataset.
    - final_test (pd.DataFrame): Test dataset.
    - meteo (pd.DataFrame): Meteorological data containing 'date', 'numer_sta', and the feature to be added.
    - feature (str): Name of the meteorological column to be added (e.g., 'ff' for wind, 'pres' for pressure).
    - index_col (str): Column used to group missing values (e.g., 'nom_reg', 'numer_sta').
    - rename_suffix (str): Suffix for renaming the column after pivoting (e.g., '_u', '_snow', '_wind', '_pres').
    - train_date (pd.DataFrame): Dates for training.
    - test_date (pd.DataFrame): Dates for testing.

    Returns:
    - final_train (pd.DataFrame): Updated training dataset.
    - final_test (pd.DataFrame): Updated test dataset.
    """
    trai = pd.concat([train_date[0:-10], final_train], axis=1)
    tes = pd.concat([test_date[:], final_test], axis=1)

    meteo_feature = meteo[['date', feature, index_col]]

    # Remplacement des valeurs manquantes par la médiane régionale
    median_by_region = meteo_feature.groupby(index_col)[feature].median()
    meteo_feature[feature] = meteo_feature.apply(
        lambda row: median_by_region[row[index_col]] if pd.isna(row[feature]) else row[feature], axis=1
    )

    # Mise sous forme de tableau pivoté
    meteo_feature = pd.pivot_table(meteo_feature, values=feature, index='date', columns=index_col)
    meteo_feature = meteo_feature.resample('30min').interpolate(method='linear', limit_direction='both')
    meteo_feature = meteo_feature.rename(columns=lambda col: f"{col}_{rename_suffix}")

    # Séparation en train et test
    train_feature = meteo_feature.loc['2017-02-13 00:30:00+00:00':'2021-12-31 22:30:00+00:00'].reset_index()
    test_feature = meteo_feature.loc['2021-12-31 22:30:00+00:00':].reset_index()

    # Fusion avec les datasets d'entraînement et de test
    final_train = trai.merge(train_feature[0:-10], how='right', on=['date'])
    final_test = tes.merge(test_feature[:], how='right', on=['date'])

    final_train.drop(columns='date', inplace=True)
    final_test.drop(columns='date', inplace=True)

    return final_train, final_test


######################################################
#### Fonction lisage des températures
######################################################

def smooth_temperature_series(df, window_size=7, poly_order=2):
    """
    Smooths all temperature columns in a DataFrame using Savitzky-Golay filter and add it to a DataFrame.

    Args:
        df (pd.DataFrame): The original DataFrame containing temperature columns.
        window_size (int): The window size for smoothing (must be odd).
        poly_order (int): Polynomial order for Savitzky-Golay filter.

    Returns:
        pd.DataFrame: The DataFrame with new smoothed temperature columns.
    """
    df_smoothed = df.copy()

    # Sélection des colonnes se terminant par '_temp'
    df_temp = df.loc[:, df.columns.str.endswith("_temp")]

    for col in df_temp.columns:
        smoothed_values = savgol_filter(df[col], window_size, poly_order)
        df_smoothed[f"{col}_smoothed"] = smoothed_values  # Ajout des colonnes lissées

    return df_smoothed

# Extracting temporal features
data_train = date_data(data_train)
data_test = date_data(data_test)
train = data_train.copy()
test = data_test.copy()
final_train_date = train['date']
final_test_date = test['date']

target_columns = ['France', 'Auvergne-Rhône-Alpes', 'Bourgogne-Franche-Comté', 'Bretagne',
       'Centre-Val de Loire', 'Grand Est', 'Hauts-de-France', 'Normandie',
       'Nouvelle-Aquitaine', 'Occitanie', 'Pays de la Loire',
       "Provence-Alpes-Côte d'Azur", 'Île-de-France',
       'Montpellier Méditerranée Métropole', 'Métropole Européenne de Lille',
       'Métropole Grenoble-Alpes-Métropole', "Métropole Nice Côte d'Azur",
       'Métropole Rennes Métropole', 'Métropole Rouen Normandie',
       "Métropole d'Aix-Marseille-Provence", 'Métropole de Lyon',
       'Métropole du Grand Nancy', 'Métropole du Grand Paris',
       'Nantes Métropole', 'Toulouse Métropole']

"""
final_train, final_test = add_previous_year_values(final_train = train, final_test = test, target_columns = target_columns)

nan_indices = final_train[final_train.isna().any(axis=1)].index
for col in final_train.columns:
    if final_train[col].isna().any():  # Vérifier s'il y a des NaN dans la colonne
        for idx in nan_indices:
            prev_value = final_train[col].iloc[idx - 2] if idx > 0 else np.nan
            next_value = final_train[col].iloc[idx + 2] if idx < len(final_train) - 1 else np.nan

            # Remplacement uniquement si les deux valeurs existent
            if not np.isnan(prev_value) and not np.isnan(next_value):
                final_train.at[idx, col] = (prev_value + next_value) / 2

"""

temp = meteo[['date', 't', 'numer_sta']]

temp = pd.pivot_table(temp, values = 't', index = 'date', columns = 'numer_sta')
temp = temp.resample('30min').interpolate(method = 'linear', limit_direction = 'both')
temp = temp.rename(columns=lambda col: col + "_temp")

# Dividing
train_temp = temp.loc['2017-02-13 00:30:00+00:00':'2021-12-31 22:30:00+00:00'].reset_index()
train_date = train_temp['date']
test_temp = temp.loc['2021-12-31 22:30:00+00:00':].reset_index()
test_date = test_temp['date']
# Merging temperature.
final_train = train.merge(train_temp, how = 'left', on = ['date'])
final_test = test.merge(test_temp, how = 'left', on = ['date'])
final_test.interpolate(method = 'linear', limi_direction = 'both', inplace = True)


final_train.drop(columns = 'date', inplace = True)
final_test.drop(columns = 'date', inplace = True)


# Utilisation de la fonction pour ajouter le vent et la pression

#final_train, final_test = add_meteo_feature(final_train, final_test, meteo, 'ff', rename_suffix = 'wind', train_date = train_date, test_date = test_date, index_col='numer_sta')
#final_train, final_test = add_meteo_feature(final_train, final_test, meteo, 'u', rename_suffix = 'humi', train_date = train_date, test_date = test_date, index_col='numer_sta')
#final_train, final_test = add_meteo_feature(final_train, final_test, meteo, 'ht_neige', rename_suffix = 'snow', train_date = train_date, test_date = test_date, index_col='numer_sta')
#final_train, final_test = add_meteo_feature(final_train, final_test, meteo, 'pres', rename_suffix = 'pres', train_date = train_date, test_date = test_date, index_col='numer_sta')
final_train, final_test = add_meteo_feature(final_train, final_test, meteo, 'td', rename_suffix = 'td', train_date = train_date, test_date = test_date, index_col='numer_sta')
#final_train, final_test = add_meteo_feature(final_train, final_test, meteo, 'tc', rename_suffix = 'tc', train_date = train_date, test_date = test_date, index_col='numer_sta')
final_train, final_test = add_meteo_feature(final_train, final_test, meteo, 'tn12', rename_suffix = 'tn12', train_date = train_date, test_date = test_date, index_col='numer_sta')
#final_train, final_test = add_meteo_feature(final_train, final_test, meteo, 'tx12', rename_suffix = 'tx12', train_date = train_date, test_date = test_date, index_col='numer_sta')


"""
final_train = smooth_temperature_series(df=final_train, window_size=7, poly_order=2)
final_test = smooth_temperature_series(df=final_test, window_size=7, poly_order=2)


df = final_train
df_smoothed = df.copy()

# Sélection des colonnes se terminant par '_temp'
df_temp = df.loc[:, df.columns.str.endswith("_temp")]

for col in df_temp.columns:
    df_smoothed.drop(columns = col)

final_train = df_smoothed """

"""### Data visualization after feature engineering

#### Average weekly consumption
"""
"""
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
weekly_consumption_france = final_train.groupby('dayofweek')['France'].mean()
weekly_consumption_bretagne = final_train.groupby('dayofweek')['Bretagne'].mean()
weekly_consumption_toulouse = final_train.groupby('dayofweek')['Toulouse Métropole'].mean()

# Plot France consumption
axes[0].plot(weekly_consumption_france.index, weekly_consumption_france.values, marker='o')
axes[0].set_xlabel('Day of the Week (0=Monday, 6=Sunday)')
axes[0].set_ylabel('Average Consumption')
axes[0].set_title('France Consumption')
axes[0].set_xticks(range(7))
axes[0].grid(True)

# Plot Bretagne consumption
axes[1].plot(weekly_consumption_bretagne.index, weekly_consumption_bretagne.values, marker='o', color='orange')
axes[1].set_xlabel('Day of the Week (0=Monday, 6=Sunday)')
axes[1].set_ylabel('Average Consumption')
axes[1].set_title('Bretagne Consumption')
axes[1].set_xticks(range(7))
axes[1].grid(True)

# Plot Toulouse consumption
axes[2].plot(weekly_consumption_toulouse.index, weekly_consumption_toulouse.values, marker='o', color='green')
axes[2].set_xlabel('Day of the Week (0=Monday, 6=Sunday)')
axes[2].set_ylabel('Average Consumption')
axes[2].set_title('Toulouse Consumption')
axes[2].set_xticks(range(7))
axes[2].grid(True)

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
"""
"""#### Average dayly consumption"""

"""
fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': 'polar'})

def plot_circular(ax, df, region, title):
    # Convertir en angles
    angles = np.arctan2(df['hour_sin'], df['hour_cos'])
    consumption = df[region]

    # Scatter plot des données horaires
    ax.scatter(angles, consumption, c=consumption, cmap='viridis', alpha=0.5, edgecolors='k', label='Hourly data')

    # Moyenne par mois et heure
    df['month'] = df['date'].dt.month
    monthly_means = df.groupby(['month', df['date'].dt.hour])[region].mean().reset_index()

    # Ajouter les coordonnées polaires
    monthly_means['hour_sin'] = np.sin(2 * np.pi * monthly_means['date'] / 24)
    monthly_means['hour_cos'] = np.cos(2 * np.pi * monthly_means['date'] / 24)
    monthly_means['angle'] = np.arctan2(monthly_means['hour_sin'], monthly_means['hour_cos'])

    # Tracer séparément chaque mois
    for month in monthly_means['month'].unique():
        subset = monthly_means[monthly_means['month'] == month].sort_values(by='angle')
        ax.plot(subset['angle'], subset[region], linestyle='-', marker='o', label=f'Mean month {month}')

    ax.set_title(title, pad=20)

    # Fixer les ticks horaires
    ax.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False))
    ax.set_xticklabels(['00h', '03h', '06h', '09h', '12h', '15h', '18h', '21h'])

    ax.set_ylim(0, max(consumption) * 1.1)
    ax.grid(True)
    ax.legend(loc='upper right', fontsize=8)

# Appliquer aux trois régions
plot_circular(axes[0], data_train, 'France', 'France Hourly Consumption')
plot_circular(axes[1], data_train, 'Bretagne', 'Bretagne Hourly Consumption')
plot_circular(axes[2], data_train, 'Toulouse Métropole', 'Toulouse Hourly Consumption')

plt.title("2021 Daily Cycle")
plt.tight_layout()
plt.show()
"""
"""#### Temperature distribution in the test set versus in the train set"""
"""
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

def plot_temperature_histogram(df, color, col, ax, label, title):
    df[col].plot(kind='hist', bins=30, color=color, ax=ax, alpha=0.7, label = label)  # Assuming 'France' is the temperature column
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Frequency')
    ax.set_title(title)

plot_temperature_histogram(final_train, 'blue', '07149_temp', axes[0], label='Final Train', title = 'Temperature Distribution Île-de-France')
plot_temperature_histogram(final_train, 'blue', '07621_temp', axes[1], label='Final Train', title = 'Temperature Distribution Occitanie')
plot_temperature_histogram(final_train, 'blue', '07020_temp', axes[2], label='Final Train', title = 'Temperature Distribution Normandie')

plot_temperature_histogram(final_test, 'red', '07149_temp', axes[0], label='Final Test', title = 'Temperature Distribution Île-de-France')
plot_temperature_histogram(final_test, 'red', '07621_temp', axes[1], label='Final Test', title = 'Temperature Distribution Occitanie')
plot_temperature_histogram(final_test, 'red', '07020_temp', axes[2], label='Final Test', title = 'Temperature Distribution Normandie')

plt.tight_layout()
plt.legend()
plt.show()
"""
"""#### Comsuption versus temperature in different regions"""
"""
data_2021 = final_train[final_train['year'] == 2021]

plt.figure(figsize=(10, 6))

# Plot Ile-de-France (in the back)
plt.plot(data_2021['07149_temp'], data_2021['Île-de-France'], 'x', alpha=0.5, label='Île-de-France')

# Plot Occitanie (middle)
plt.plot(data_2021['07621_temp'], data_2021['Occitanie'], 'x', alpha=0.7, label='Occitanie')

# Plot Normandie (front)
plt.plot(data_2021['07020_temp'], data_2021['Normandie'], 'x', label='Normandie')


plt.xlabel('Temperature')
plt.ylabel('Consommation')
plt.title('Comsuption vs Temperature (2021)')
plt.legend()
plt.grid(True)
plt.show()
"""
"""# Deep Learning Models"""
"""## Custom Loss"""
class CustomLoss(nn.Module):
    """
    Custom loss function based on RMSE, with specific weights for France's electricity consumption,
    regional consumption, and seasonal adjustments (summer, winter).

    Attributes:
    -----------
    - france_weight (float): Weight applied to the prediction of France's total consumption.
    - region_weight (float): Weight applied to the prediction of 12 regional consumption values.
    - summer_weight (float): Weight factor for errors made in summer.
    - winter_weight (float): Weight factor for errors made in winter.

    Methods:
    --------
    - rmse(y_pred, y_true):
        Computes the Root Mean Squared Error (RMSE) between predictions and actual values.

    - forward(y_pred, y_true, season):
        Computes the total loss by applying different weightings based on the target variable
        (France, region, or metropolis) and the season.

    Parameters:
    -----------
    - y_pred (torch.Tensor): Tensor of model predictions, shape (batch_size, num_targets).
    - y_true (torch.Tensor): Tensor of actual values, shape (batch_size, num_targets).
    - season (torch.Tensor): Tensor indicating the season for each sample (batch_size,).

    Returns:
    --------
    - total_loss (torch.Tensor): Total weighted loss value.
    """

    def __init__(self, france_weight=2.0, region_weight=1.0, summer_weight=1.5, winter_weight=2.0):
        super(CustomLoss, self).__init__()
        self.france_weight = france_weight
        self.region_weight = region_weight
        self.summer_weight = summer_weight
        self.winter_weight = winter_weight

    def rmse(self, y_pred, y_true):
        """
        Computes the Root Mean Squared Error (RMSE) between predictions and actual values.

        Parameters:
        -----------
        - y_pred (torch.Tensor): Tensor of predictions.
        - y_true (torch.Tensor): Tensor of actual values.

        Returns:
        --------
        - rmse (torch.Tensor): RMSE error between y_pred and y_true.
        """
        mse = torch.mean((y_pred - y_true) ** 2)
        return torch.sqrt(mse)


    def forward(self, y_pred, y_true, saison):
        """
        Computes the total loss by applying weight adjustments based on the target variable and season.

        Parameters:
        -----------
        - y_pred (torch.Tensor): Tensor of model predictions.
        - y_true (torch.Tensor): Tensor of actual values.
        - season (torch.Tensor): Tensor of seasons (1 = winter, 3 = summer).

        Returns:
        --------
        - total_loss (torch.Tensor): Total loss value.
        """
        total_loss = 0.0
        for i in range(y_pred.shape[1]):
            # Déterminer le poids selon si la colonne représente la France, une région ou une métropole
            if i == 0:
                weight = self.france_weight  # France
            elif 1 <= i <= 12:
                weight = self.region_weight  # Régions
            else:
                weight = 1.0  # Métropoles (poids neutre)

            # Appliquer une pondération saisonnière
            seasonal_weight = torch.where(
                (saison == 3), self.summer_weight,  # Été
                torch.where((saison == 1), self.winter_weight, 1.0)  # Hiver sinon poids normal
            )

            # Calcul de la RMSE pondérée
            loss = self.rmse(y_pred[:, i], y_true[:, i]) * seasonal_weight
            total_loss += torch.mean(loss * weight)

        return total_loss

def MLP_train_test_loop(df, target_columns, epochs=200, use_batches=True, batch_size=4096,
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                    model=None, criterion=None, optimizer=None, train_ratio=0.85,
                    categorical_features=['month', 'day', 'dayofweek', 'saison'], data_pred= "Data/pred.csv" ):
    """
    Trains a Multi-Layer Perceptron (MLP) model, evaluates its performance, and makes predictions on new data.

    This function performs:
    - Data preprocessing: One-hot encoding categorical features and normalizing numerical data.
    - Training a neural network model using PyTorch, with optional mini-batch processing.
    - Evaluating the model using RMSE.
    - Visualizing predicted vs actual values.
    - Making final predictions on new data and saving the results.

    Parameters:
    -----------
    - df (pd.DataFrame): The input dataset containing features and target variables.
    - target_columns (list): List of target column names.
    - epochs (int, optional): Number of training epochs (default is 200).
    - use_batches (bool, optional): Whether to use mini-batch training (default is True).
    - batch_size (int, optional): Size of each mini-batch if batching is enabled (default is 4096).
    - device (torch.device, optional): The device (CPU or GPU) to use for model training (default is GPU if available).
    - model (torch.nn.Module, optional): The neural network model to use (default is None, requiring an initialized model).
    - criterion (torch.nn.Module, optional): The loss function (default is None, using a custom RMSE-based loss).
    - optimizer (torch.optim.Optimizer, optional): The optimizer to use (default is Adam with a learning rate of 0.005).
    - train_ratio (float, optional): Ratio of data to use for training (default is 85%).
    - categorical_features (list, optional): List of categorical feature column names (default includes 'month', 'day', etc.).
    - data_pred (str, optional): Path to the dataset used for predictions (default is "Data/pred.csv").

    Returns:
    --------
    - model (torch.nn.Module): The trained MLP model.
    - pred (pd.DataFrame): DataFrame containing predictions saved as "pred2.csv".

    Notes:
    ------
    - The function automatically handles missing values by normalizing and encoding categorical variables.
    - Training is performed using either mini-batch or full dataset optimization.
    - RMSE is computed and displayed for each target column.
    - A plot is generated to compare actual vs predicted values for the first target column.
    - Final predictions are saved in "pred2.csv".
    """

    # Séparer les features et les cibles
    X = df.drop(columns=target_columns).copy()
    y = df[target_columns].values

    # Encodage OneHotEncoder des variables catégoriques
    enc = OneHotEncoder(handle_unknown='ignore')
    encoded_features = enc.fit_transform(X[categorical_features]).toarray()
    encoded_df = pd.DataFrame(encoded_features, columns=enc.get_feature_names_out(categorical_features))

    # Concaténation des features encodées avec les autres colonnes
    X = pd.concat([X.drop(columns=categorical_features).reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

    # Normalisation
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y)

    # Division en train/test (85% train, 15% test)
    split_index = int(train_ratio * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    saison_train, saison_test = df['saison'].values[:split_index], df['saison'].values[split_index:]

    # Initialisation du modèle, de la loss et de l'optimiseur

    model = model(input_dim=X.shape[1], output_dim=y.shape[1]).to(device)
    if criterion == None :
        criterion = CustomLoss(france_weight=2.5, region_weight=1.5, summer_weight=1.5, winter_weight=2.0)
    if optimizer == None :
        optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)

    if use_batches == True :
        # Conversion en tenseurs PyTorch
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)
        saison_train = torch.tensor(saison_train, dtype=torch.float32)
        saison_test = torch.tensor(saison_test, dtype=torch.float32)

        # Création des DataLoader
        train_dataset = TensorDataset(X_train, y_train, saison_train)
        test_dataset = TensorDataset(X_test, y_test, saison_test)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for batch_X, batch_y, batch_saison in train_loader:
                batch_X, batch_y, batch_saison = batch_X.to(device), batch_y.to(device), batch_saison.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y, batch_saison)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss / len(train_loader):.4f}')

        model.eval()
        y_pred = []
        y_test = []

        with torch.no_grad():
            for batch_X, batch_y, saison_batch in test_loader:
                batch_X, batch_y, saison_batch = batch_X.to(device), batch_y.to(device), saison_batch.to(device)

                batch_pred = model(batch_X).cpu().numpy()
                y_pred.append(batch_pred)
                y_test .append(batch_y.cpu().numpy())

        y_pred = np.vstack(y_pred)
        y_test  = np.vstack(y_test)

        # Inverser la normalisation des prédictions et des vraies valeurs
        y_pred = scaler_y.inverse_transform(y_pred)
        y_test  = scaler_y.inverse_transform(y_test)

    else :
        # Conversion en tenseurs PyTorch
        X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
        saison_train = torch.tensor(saison_train, dtype=torch.int64).to(device)
        saison_test = torch.tensor(saison_test, dtype=torch.int64).to(device)

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train, saison_train)
            loss.backward()
            optimizer.step()
            if (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

        model.eval()
        y_pred = model(X_test).detach().cpu().numpy()
        y_pred = scaler_y.inverse_transform(y_pred)
        y_test = scaler_y.inverse_transform(y_test.cpu().numpy())


    # Calcul des RMSE pour chaque colonne
    rmse_per_column = np.sqrt(np.mean((y_pred - y_test) ** 2, axis=0))
    sum_rmse = np.sum(rmse_per_column)

    for i, col_name in enumerate(target_columns):
        print(f'RMSE pour {col_name}: {rmse_per_column[i]:.4f}')
    print(f'🔥 Somme des RMSE: {sum_rmse:.4f}')

    """
    # Affichage du graphe pour la première colonne
    plt.figure(figsize=(10, 5))
    plt.plot(y_test[:, 0], label='Valeurs Réelles', color='blue')
    plt.plot(y_pred[:, 0], label='Valeurs Prédites', color='red', linestyle='dashed')
    plt.legend()
    plt.xlabel('Échantillons')
    plt.ylabel('Consommation électrique')
    plt.title(f'Comparaison des valeurs réelles et prédites ({target_columns[0]})')
    plt.show()
    """
    joblib.dump(scaler_X, "scaler_X.joblib")
    joblib.dump(scaler_y, "scaler_y.joblib")
    return model, y_test, y_pred
def CMAtt_train_test_loop(df, target_columns, epochs=200, use_batches=True, batch_size=4096,
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                    model=None, criterion=None, optimizer=None, train_ratio=0.85,
                    categorical_features=['month', 'day', 'dayofweek', 'saison'], data_pred= "Data/pred.csv" ):
    """
    Trains a Multi-Layer Perceptron (MLP) model, evaluates its performance, and makes predictions on new data.

    This function performs:
    - Data preprocessing: One-hot encoding categorical features and normalizing numerical data.
    - Training a neural network model using PyTorch, with optional mini-batch processing.
    - Evaluating the model using RMSE.
    - Visualizing predicted vs actual values.
    - Making final predictions on new data and saving the results.

    Parameters:
    -----------
    - df (pd.DataFrame): The input dataset containing features and target variables.
    - target_columns (list): List of target column names.
    - epochs (int, optional): Number of training epochs (default is 200).
    - use_batches (bool, optional): Whether to use mini-batch training (default is True).
    - batch_size (int, optional): Size of each mini-batch if batching is enabled (default is 4096).
    - device (torch.device, optional): The device (CPU or GPU) to use for model training (default is GPU if available).
    - model (torch.nn.Module, optional): The neural network model to use (default is None, requiring an initialized model).
    - criterion (torch.nn.Module, optional): The loss function (default is None, using a custom RMSE-based loss).
    - optimizer (torch.optim.Optimizer, optional): The optimizer to use (default is Adam with a learning rate of 0.005).
    - train_ratio (float, optional): Ratio of data to use for training (default is 85%).
    - categorical_features (list, optional): List of categorical feature column names (default includes 'month', 'day', etc.).
    - data_pred (str, optional): Path to the dataset used for predictions (default is "Data/pred.csv").

    Returns:
    --------
    - model (torch.nn.Module): The trained MLP model.
    - pred (pd.DataFrame): DataFrame containing predictions saved as "pred2.csv".

    Notes:
    ------
    - The function automatically handles missing values by normalizing and encoding categorical variables.
    - Training is performed using either mini-batch or full dataset optimization.
    - RMSE is computed and displayed for each target column.
    - A plot is generated to compare actual vs predicted values for the first target column.
    - Final predictions are saved in "pred2.csv".
    """

    # Séparer les features et les cibles
    X = df.drop(columns=target_columns).copy()
    y = df[target_columns].values

    # Encodage OneHotEncoder des variables catégoriques
    enc = OneHotEncoder(handle_unknown='ignore')
    encoded_features = enc.fit_transform(X[categorical_features]).toarray()
    encoded_df = pd.DataFrame(encoded_features, columns=enc.get_feature_names_out(categorical_features))

    # Concaténation des features encodées avec les autres colonnes
    X = pd.concat([X.drop(columns=categorical_features).reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

    # Normalisation
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y)
    
    print("Scalers saved successfully!")
    # Division en train/test (85% train, 15% test)
    split_index = int(train_ratio * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    saison_train, saison_test = df['saison'].values[:split_index], df['saison'].values[split_index:]

    # Initialisation du modèle, de la loss et de l'optimiseur

    model = model( output_dim=y.shape[1]).to(device)
    if criterion == None :
        criterion = CustomLoss(france_weight=2.5, region_weight=1.5, summer_weight=1.5, winter_weight=2.0)
    if optimizer == None :
        optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)

    if use_batches == True :
        # Conversion en tenseurs PyTorch
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)
        saison_train = torch.tensor(saison_train, dtype=torch.float32)
        saison_test = torch.tensor(saison_test, dtype=torch.float32)

        # Création des DataLoader
        train_dataset = TensorDataset(X_train, y_train, saison_train)
        test_dataset = TensorDataset(X_test, y_test, saison_test)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for batch_X, batch_y, batch_saison in train_loader:
                batch_X, batch_y, batch_saison = batch_X.to(device), batch_y.to(device), batch_saison.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y, batch_saison)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss / len(train_loader):.4f}')

        model.eval()
        y_pred = []
        y_test = []

        with torch.no_grad():
            for batch_X, batch_y, saison_batch in test_loader:
                batch_X, batch_y, saison_batch = batch_X.to(device), batch_y.to(device), saison_batch.to(device)

                batch_pred = model(batch_X).cpu().numpy()
                y_pred.append(batch_pred)
                y_test .append(batch_y.cpu().numpy())

        y_pred = np.vstack(y_pred)
        y_test  = np.vstack(y_test)

        # Inverser la normalisation des prédictions et des vraies valeurs
        y_pred = scaler_y.inverse_transform(y_pred)
        y_test  = scaler_y.inverse_transform(y_test)

    else :
        # Conversion en tenseurs PyTorch
        X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
        saison_train = torch.tensor(saison_train, dtype=torch.int64).to(device)
        saison_test = torch.tensor(saison_test, dtype=torch.int64).to(device)

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train, saison_train)
            loss.backward()
            optimizer.step()
            if (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

        model.eval()
        y_pred = model(X_test).detach().cpu().numpy()
        y_pred = scaler_y.inverse_transform(y_pred)
        y_test = scaler_y.inverse_transform(y_test.cpu().numpy())


    # Calcul des RMSE pour chaque colonne
    rmse_per_column = np.sqrt(np.mean((y_pred - y_test) ** 2, axis=0))
    sum_rmse = np.sum(rmse_per_column)

    for i, col_name in enumerate(target_columns):
        print(f'RMSE pour {col_name}: {rmse_per_column[i]:.4f}')
    print(f'🔥 Somme des RMSE: {sum_rmse:.4f}')
    """
    # Affichage du graphe pour la première colonne
    plt.figure(figsize=(10, 5))
    plt.plot(y_test[:, 0], label='Valeurs Réelles', color='blue')
    plt.plot(y_pred[:, 0], label='Valeurs Prédites', color='red', linestyle='dashed')
    plt.legend()
    plt.xlabel('Échantillons')
    plt.ylabel('Consommation électrique')
    plt.title(f'Comparaison des valeurs réelles et prédites ({target_columns[0]})')
    plt.show()
    """
    joblib.dump(scaler_X, "scaler_X.pkl")
    joblib.dump(scaler_y, "scaler_y.pkl")
    return model, y_test, y_pred

final_test.to_csv("final_test.csv",index=False)
final_train.to_csv("final_train.csv",index=False)
    

    


"""# Training and predictions"""
if __name__ == "__main__":
    # Vérification et utilisation du GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Utilisation de {device}')

    target_columns = ['France', 'Auvergne-Rhône-Alpes', 'Bourgogne-Franche-Comté', 'Bretagne',
        'Centre-Val de Loire', 'Grand Est', 'Hauts-de-France', 'Normandie',
        'Nouvelle-Aquitaine', 'Occitanie', 'Pays de la Loire',
        "Provence-Alpes-Côte d'Azur", 'Île-de-France',
        'Montpellier Méditerranée Métropole', 'Métropole Européenne de Lille',
        'Métropole Grenoble-Alpes-Métropole', "Métropole Nice Côte d'Azur",
        'Métropole Rennes Métropole', 'Métropole Rouen Normandie',
        "Métropole d'Aix-Marseille-Provence", 'Métropole de Lyon',
        'Métropole du Grand Nancy', 'Métropole du Grand Paris',
        'Nantes Métropole', 'Toulouse Métropole']

    seed = 13

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(final_train.shape)

    print("Training MLP model...")
    model = MLP_Model
    criterion = CustomLoss(france_weight=2.5, region_weight=1.5, summer_weight=1.5, winter_weight=2.0)


    model, y_test, y_pred = MLP_train_test_loop(df = final_train, target_columns = target_columns, epochs=200, use_batches=False, batch_size=4096,
                        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                        model=model, criterion=criterion, optimizer=None, train_ratio=0.85,
                        categorical_features=['month', 'day', 'dayofweek', 'saison'], data_pred = "Data/pred.csv")

    torch.save(model.state_dict(), "MLP_model.pth")  # Save trained model
    print("MLP Training completed and model saved!")

    print("Training Batch MLP model...")
    '''
    model = MLP_Batch_Model
    criterion = CustomLoss(france_weight=2.5, region_weight=1.5, summer_weight=1.5, winter_weight=2.0)


    model,_,_= MLP_train_test_loop(df = final_train, target_columns = target_columns, epochs=200, use_batches=True, batch_size=4096,
                        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                        model=model, criterion=criterion, optimizer=None, train_ratio=0.99,
                        categorical_features=['month', 'day', 'dayofweek', 'saison'], data_pred = "Data/pred.csv" )

    torch.save(model.state_dict(), "MLP_batch_model.pth")  # Save trained model
    print("Batch MLP Training completed and model saved!")
    '''
    """
    print("Training CNN model...")
    model = CNNModel
    criterion = CustomLoss(france_weight=2.5, region_weight=1.5, summer_weight=1.5, winter_weight=2.0)
    model, y_test, y_pred = CMAtt_train_test_loop(df = final_train, target_columns = target_columns, epochs=200, use_batches=True, batch_size=4096,
                        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                        model=model, criterion=criterion, optimizer=None, train_ratio=0.99,
                        categorical_features=['month', 'day', 'dayofweek', 'saison'], data_pred = "Data/pred.csv" )

    torch.save(model.state_dict(), "CNNModel.pth")  # Save trained model
    print("CNN Training completed and model saved!")


    print("Training CNN Attention model...")
    model = CNN_Attention
    criterion = CustomLoss(france_weight=2.5, region_weight=1.5, summer_weight=1.5, winter_weight=2.0)


    model, y_test, y_pred = CMAtt_train_test_loop(df = final_train, target_columns = target_columns, epochs=200, use_batches=True, batch_size=4096,
                        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                        model=model, criterion=criterion, optimizer=None, train_ratio=0.99,
                        categorical_features=['month', 'day', 'dayofweek', 'saison'], data_pred = "Data/pred.csv" )

    torch.save(model.state_dict(), "CNN_Attention.pth")  # Save trained model
    print("CNN Attention Training completed and model saved!")
    """






    
    print ("All the trainings are done, predict using predict.py")
    