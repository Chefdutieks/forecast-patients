# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 16:49:01 2024

@author: alice

Ce fichier utilise le modèle de prédiction pour prédire le nombre de patients dans le futur.
Il récupère les données de clients actifs depuis Google Sheets, génère des données futures, effectue des prédictions et écrit les résultats de nouveau dans Google Sheets.
"""

import pandas as pd
import numpy as np
import holidays
from vacances_scolaires_france import SchoolHolidayDates
import random_forest_model
from googleapiclient.discovery import build
from google.oauth2 import service_account
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')

# Colonnes utilisées dans le modèle initial
MODEL_COLUMNS = [
    'ActiveClients', 'Holiday', 'AfterHoliday', 'SchoolHoliday', 'WeekOfYear', 'Hour_9', 'Hour_10', 'Hour_11', 'Hour_12',
    'Hour_13', 'Hour_14', 'Hour_15', 'Hour_16', 'Hour_17', 'Hour_18', 'Hour_19',
    'Minute_30', 'DayOfWeek_1', 'DayOfWeek_2', 'DayOfWeek_3', 'DayOfWeek_4', 'DayOfWeek_5', 'DayOfWeek_6',
    'DayOfMonth_2', 'DayOfMonth_3', 'DayOfMonth_4', 'DayOfMonth_5', 'DayOfMonth_6', 'DayOfMonth_7', 'DayOfMonth_8', 'DayOfMonth_9',
    'DayOfMonth_10', 'DayOfMonth_11', 'DayOfMonth_12', 'DayOfMonth_13', 'DayOfMonth_14', 'DayOfMonth_15', 'DayOfMonth_16', 'DayOfMonth_17',
    'DayOfMonth_18', 'DayOfMonth_19', 'DayOfMonth_20', 'DayOfMonth_21', 'DayOfMonth_22', 'DayOfMonth_23', 'DayOfMonth_24', 'DayOfMonth_25',
    'DayOfMonth_26', 'DayOfMonth_27', 'DayOfMonth_28', 'DayOfMonth_29', 'DayOfMonth_30', 'DayOfMonth_31',
    'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6', 'Month_7', 'Month_8', 'Month_9', 'Month_10', 'Month_11', 'Month_12','Year_2023','Year_2024', 
    'Year_2025'
]

def read_google_sheet(sheet_name, range_name):
    """
    Lit les données depuis une feuille Google Sheets.

    Args:
        sheet_name (str): Nom de la feuille.
        range_name (str): Plage de cellules à lire.

    Returns:
        DataFrame: Données lues depuis Google Sheets sous forme de DataFrame.
    """
    SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
    SERVICE_ACCOUNT_FILE = 'credentials.json'
    
    creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    
    SAMPLE_SPREADSHEET_ID = "16tnjKRHw6nzhV-HiLxX0j99XZiyFkjvCc9804VXJexA"
    
    service = build("sheets", "v4", credentials=creds)
    
    # Lire les données existantes depuis Google Sheets
    result = service.spreadsheets().values().get(spreadsheetId=SAMPLE_SPREADSHEET_ID, range=f"{sheet_name}!{range_name}").execute()
    existing_data = result.get("values", [])
    
    # Convertir les données existantes en DataFrame
    existing_df = pd.DataFrame(existing_data[1:], columns=existing_data[0])
    
    return existing_df

def process_active_clients(existing_df):
    """
    Traite les données des clients actifs à partir du DataFrame récupéré depuis le google sheet.

    Args:
        existing_df (DataFrame): Données existantes.

    Returns:
        dict: Clients actifs par mois.
    """
    month_map = {
        'janvier': 1,
        'février': 2,
        'mars': 3,
        'avril': 4,
        'mai': 5,
        'juin': 6,
        'juillet': 7,
        'août': 8,
        'septembre': 9,
        'octobre': 10,
        'novembre': 11,
        'décembre': 12,
        'fevrier': 2,  # pour les mois remplacés
        'aout': 8,
        'decembre': 12
    }
    
    def convert_to_datetime(row):
        month_str, year_str = row.split()
        month = month_map.get(month_str.lower())
        if month:
            return pd.Timestamp(year=int(year_str), month=month, day=1)
        return pd.NaT
    
    existing_df['Month'] = existing_df['Month'].apply(convert_to_datetime)
    
    existing_df['ActiveClients'] = existing_df['ActiveClients'].astype(int)
    existing_df['PredictedActiveClients'] = existing_df['PredictedActiveClients'].astype(int)
    
    active_clients_per_month = {
        (row['Month'].year, row['Month'].month): row['PredictedActiveClients']
        for _, row in existing_df.iterrows() if not pd.isnull(row['Month'])
    }
    
    return active_clients_per_month

def generate_future_data(start_date_str, end_date_str, active_clients_per_month):
    """
    Génère les données futures (sans les patients) basées sur les clients actifs et les dates fournies.

    Args:
        start_date_str (str): Date de début.
        end_date_str (str): Date de fin.
        active_clients_per_month (dict): Clients actifs par mois.

    Returns:
        DataFrame: Données futures générées.
    """
    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)
    date_range = pd.date_range(start_date, end_date, freq='D')
    hours = list(range(8, 20))
    minutes = [0, 30]
    future_data = []

    for date in date_range:
        active_clients = active_clients_per_month.get((date.year, date.month), 0)
        for hour in hours:
            for minute in minutes:
                future_data.append({
                    'Day': date,
                    'Hour': hour,
                    'Minute': minute,
                    'ActiveClients': active_clients
                })
    
    return pd.DataFrame(future_data)

def data_preprocessing(data):
    """
    Transforme les données afin qu'elles aient la même structure qu'ilisée pour la construction du modèle.

    Args:
        data (DataFrame): Données brutes à prétraiter.

    Returns:
        DataFrame: Données prétraitées avec les colonnes d'origine pour référence.
    """
    # Convertir la colonne 'Day' en datetime
    data['Day'] = pd.to_datetime(data['Day'])
    
    # Déterminer les jours fériés
    all_holidays = holidays.France(years=range(2020, 2031))
    holiday_dates = set(pd.to_datetime(list(all_holidays.keys())))
    
    # Ajouter les colonnes Holiday et AfterHoliday
    data['Holiday'] = data['Day'].isin(holiday_dates).astype(int)
    data['AfterHoliday'] = data['Day'].shift(24).isin(holiday_dates).astype(int)
    
    # Déterminer les vacances scolaires
    school_holidays = SchoolHolidayDates()
    data['SchoolHoliday'] = data['Day'].apply(lambda x: school_holidays.is_holiday(x.date())).astype(int)
    
    # Extraire les caractéristiques temporelles
    data['DayOfWeek'] = data['Day'].dt.dayofweek
    data['DayOfMonth'] = data['Day'].dt.day
    data['Month'] = data['Day'].dt.month
    data['Year'] = data['Day'].dt.year
    data['WeekOfYear'] = data['Day'].dt.isocalendar().week.astype(int)
    
    # Convertir les colonnes temporelles en variables catégorielles
    data['Hour'] = data['Hour'].astype('category')
    data['Minute'] = data['Minute'].astype('category')
    data['DayOfWeek'] = data['DayOfWeek'].astype('category')
    data['DayOfMonth'] = data['DayOfMonth'].astype('category')
    data['Month'] = data['Month'].astype('category')
    data['Year'] = data['Year'].astype('category')
    
    # Encodage One-Hot des colonnes catégorielles
    data = pd.get_dummies(data, columns=['Hour', 'Minute', 'DayOfWeek', 'DayOfMonth', 'Month', 'Year'], drop_first=True)
    
    # Ajouter des colonnes manquantes avec des zéros
    missing_cols = set(MODEL_COLUMNS) - set(data.columns)
    for col in missing_cols:
        data[col] = 0
    
    # S'assurer que les colonnes sont dans le même ordre que MODEL_COLUMNS
    data = data[[col for col in MODEL_COLUMNS if col in data.columns]]
    
    # Préparer les variables explicatives
    X = data.drop(columns=['Day', 'Patients'], errors='ignore')
    
    return X

def generate_and_predict(start_date_str, end_date_str, active_clients_per_month, rf_model):
    """
    Génère des données futures et effectue des prédictions des patients à l'aide du modèle Random Forest.

    Args:
        start_date_str (str): Date de début.
        end_date_str (str): Date de fin.
        active_clients_per_month (dict): Clients actifs par mois.
        rf_model (RandomForestRegressor): Modèle Random Forest entraîné.

    Returns:
        DataFrame: Données futures avec les prédictions.
    """
    # Générer les données futures
    future_data = generate_future_data(start_date_str, end_date_str, active_clients_per_month)

    # Effectuer le prétraitement sur les nouvelles données
    X_future = data_preprocessing(future_data)

    # Effectuer les prédictions sur les nouvelles données
    future_predictions = rf_model.predict(X_future)

    # Arrondir les prédictions à l'entier le plus proche
    future_predictions = np.round(future_predictions).astype(int)

    # Ajouter les prédictions au DataFrame
    future_data['PredictedPatients'] = future_predictions
    # Créer une colonne WeekYear en fusionnant Year et WeekOfYear
    future_data['WeekYear'] = future_data['Year'].astype(str) + '-' + future_data['WeekOfYear'].astype(str)
    
    # Convertir la colonne 'Day' en format DD/MM/YYYY
    future_data['Day'] = future_data['Day'].dt.strftime('%d/%m/%Y')
    
    # Réorganiser les colonnes
    future_data = future_data[['WeekYear', 'Day', 'Hour', 'Minute', 'Holiday', 'ActiveClients', 'PredictedPatients']]
  
    return future_data

def write_sheet(future_data, existing_google_api_df):
    """
    Écrit les données mises à jour dans Google Sheets.

    Args:
        future_data (DataFrame): Données futures avec les prédictions.
        existing_google_api_df (DataFrame): Données existantes dans Google Sheets.
    """
    SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
    SERVICE_ACCOUNT_FILE = 'credentials.json'
    
    creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    
    SAMPLE_SPREADSHEET_ID = "16tnjKRHw6nzhV-HiLxX0j99XZiyFkjvCc9804VXJexA"
    
    service = build("sheets", "v4", credentials=creds)
    
    # Convertir les colonnes 'Day', 'Hour', et 'Minute' en types appropriés
    existing_google_api_df['Day'] = pd.to_datetime(existing_google_api_df['Day'], format='%d/%m/%Y', errors='coerce')
    existing_google_api_df['Hour'] = existing_google_api_df['Hour'].astype(int)
    existing_google_api_df['Minute'] = existing_google_api_df['Minute'].astype(int)

    # Supprimer les lignes dont la date est plus de cinq semaines avant la date actuelle
    five_weeks_ago = datetime.today() - timedelta(weeks=5)
    existing_google_api_df = existing_google_api_df[existing_google_api_df['Day'] >= five_weeks_ago]
    
    # Affichage pour vérifier les données après suppression
    print("Données après suppression des lignes trop anciennes :")
    print(existing_google_api_df)
    
    # Convertir la colonne 'Day' de future_data en datetime
    future_data['Day'] = pd.to_datetime(future_data['Day'], format='%d/%m/%Y')
    future_data['Hour'] = future_data['Hour'].astype(int)
    future_data['Minute'] = future_data['Minute'].astype(int)
    
    # Merge les nouvelles données avec les données existantes
    combined_df = pd.merge(existing_google_api_df, future_data, on=['Day', 'Hour', 'Minute'], how='outer', suffixes=('', '_new'))
    
    # Remplacer les valeurs existantes par les nouvelles valeurs si elles sont présentes
    for col in future_data.columns:
        if col != 'Day' and col != 'Hour' and col != 'Minute':
            combined_df[col] = combined_df[col + '_new'].combine_first(combined_df[col])
            combined_df.drop(columns=[col + '_new'], inplace=True)
    
    # Réorganiser les colonnes dans l'ordre souhaité
    ordered_columns = ['WeekYear', 'Day', 'Hour', 'Minute', 'Holiday', 'ActiveClients', 'PredictedPatients']
    combined_df = combined_df[ordered_columns]
    
    # Convertir les dates en chaînes de caractères pour éviter les problèmes de sérialisation JSON
    combined_df['Day'] = combined_df['Day'].dt.strftime('%d/%m/%Y')
    
    # Convertir les colonnes en listes pour l'API Google Sheets
    combined_data = [combined_df.columns.values.tolist()] + combined_df.values.tolist()
    
    # BUG JUILLET SUPPRESSION
    
    # Écrire les données mises à jour dans Google Sheets
    request = service.spreadsheets().values().update(
        spreadsheetId=SAMPLE_SPREADSHEET_ID,
        range="google api!A1",  # Ajustez cette plage en fonction de votre feuille
        valueInputOption="USER_ENTERED",
        body={"values": combined_data}
    ).execute()

    print(f"Les données ont été écrites dans Google Sheets : {request}")

def get_date_range(weeks):
    """
    Détermine la plage de dates pour la génération des données futures.
    
    Args:
        weeks (int) : nombre de semaine à prédire 
    Returns:
        tuple: Date de début et date de fin sous forme de chaînes de caractères.
    """
    start_date = datetime.today()
    end_date = start_date + timedelta(weeks=weeks)
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

