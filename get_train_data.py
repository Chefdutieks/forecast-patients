# -*- coding: utf-8 -*-
"""
Modified on Fri June 13 14:34:13 2025

@author: larrybird

Ce fichier a pour but de récupérer des données d'arrivée de patients depuis la base de données prod-kpi, et d'effectuer un traitement des valeurs manquantes sur ces données pour générer un jeu de données d'entraînement utilisable pour un modèle de prédiction.
"""

import pymssql
import pandas as pd
from vacances_scolaires_france import SchoolHolidayDates
from dotenv import load_dotenv
import holidays
import os
import re
load_dotenv()


def get_connection_config():
    return {
        'server': os.getenv('AZURE_DB_SERVER'),
        'database': os.getenv('AZURE_DB_NAME'),
        'user': os.getenv('AZURE_DB_USER'),
        'password': os.getenv('AZURE_DB_PASSWORD'),
        'port': 1433
    }


def get_query():
    return """
        WITH FilteredMatchings AS (
            SELECT 
                pm.PatientId,
                pm.ClientId,
                pm.StartedAt,
                CAST(pm.StartedAt AS date) AS Day,
                FORMAT(pm.StartedAt, 'yyyy-MM') AS Month,
                pm.hourRounded,
                c.Name AS ClientName
            FROM PatientMatchings pm
            JOIN Clients c ON pm.ClientId = c.Id
            WHERE 
                DATEPART(hour, pm.StartedAt) BETWEEN 8 AND 20
                AND c.Name NOT LIKE 'TAB%'
        ),
        PatientUnique AS (
            SELECT 
                PatientId,
                Day,
                MIN(StartedAt) AS MaxStartedAt
            FROM FilteredMatchings
            GROUP BY PatientId, Day
        ),
        PatientHours AS (
            SELECT 
                pu.PatientId,
                pu.Day,
                fm.hourRounded
            FROM PatientUnique pu
            JOIN FilteredMatchings fm 
                ON pu.PatientId = fm.PatientId AND pu.MaxStartedAt = fm.StartedAt
        ),
        ActiveClientsByDay AS (
            SELECT 
                Day,
                COUNT(DISTINCT ClientId) AS ActiveClientsDay
            FROM FilteredMatchings
            GROUP BY Day
        ),
        ActiveClientsByMonth AS (
            SELECT 
                Month,
                COUNT(DISTINCT ClientId) AS ActiveClientsMonth
            FROM FilteredMatchings
            GROUP BY Month
        )
        SELECT 
            ph.Day,
            ph.hourRounded AS Hour,
            COUNT(DISTINCT ph.PatientId) AS Patients,
            acd.ActiveClientsDay,
            acm.ActiveClientsMonth
        FROM PatientHours ph
        JOIN ActiveClientsByDay acd ON ph.Day = acd.Day
        JOIN ActiveClientsByMonth acm ON FORMAT(ph.Day, 'yyyy-MM') = acm.Month
        GROUP BY 
            ph.Day,
            ph.hourRounded,
            acd.ActiveClientsDay,
            acm.ActiveClientsMonth
        ORDER BY ph.Day, ph.hourRounded;
    """


def get_connection(config):
    return pymssql.connect(
        server=config['server'],
        port=config['port'],
        user=config['user'],
        password=config['password'],
        database=config['database'],
        login_timeout=30,
        tds_version='7.4'
    )


def execute_query(connection, query):
    return pd.read_sql(query, connection)


def data_preprocessing(data):
    """
    Effectue le prétraitement des données extraites de la base de données.
    - Convertit les colonnes 'Day' et 'Hour' en types datetime appropriés.
    - Filtre les données pour ne garder que les jours de semaine, les heures entre 08:30 et 19:30,
    et les patients non nuls.
    - Ajoute des colonnes pour les jours fériés, les vacances scolaires, le jour de la semaine,
    le mois, l'année, la semaine de l'année et la semaine du mois.
    - Effectue un encodage one-hot pour les colonnes 'Hour' et 'DayOfWeek'.
    - Renomme les colonnes pour enlever les caractères spéciaux.
    Args:
        data (pd.DataFrame): DataFrame contenant les données extraites de la base de données.
    Returns:        
        pd.DataFrame: DataFrame prétraité avec les colonnes nécessaires pour l'entraînement du modèle.
        pd.DataFrame: DataFrame original avec les colonnes de base pour référence.
    """
    data = data.copy()
    data['Day'] = pd.to_datetime(data['Day'])
    data['Hour'] = pd.to_datetime(data['Hour'], format='%H:%M').dt.time
    
    start = pd.to_datetime('08:30').time()
    end = pd.to_datetime('19:30').time()
    data = data[
        (data['Day'].dt.dayofweek != 6) &
        (data['Patients'] != 0) &
        (data['Hour'] >= start) &
        (data['Hour'] <= end)
    ].copy()

    all_holidays = holidays.country_holidays('FR', years=range(2020, 2031))
    holiday_dates = set(pd.to_datetime(list(all_holidays.keys())))
    holiday_dates.add(pd.to_datetime('2024-05-30'))

    data.loc[:, 'Holiday'] = data['Day'].isin(holiday_dates).astype(int)
    data = data.sort_values(['Day', 'Hour'])
    data.loc[:, 'AfterHoliday'] = data['Day'].shift(48).isin(holiday_dates).astype(int)

    school_holidays = SchoolHolidayDates()
    data.loc[:, 'SchoolHoliday'] = data['Day'].apply(lambda x: school_holidays.is_holiday(x.date())).astype(int)

    data['DayOfWeek'] = data['Day'].dt.dayofweek
    data['Month'] = data['Day'].dt.month
    data['Year'] = data['Day'].dt.year
    data['WeekOfYear'] = data['Day'].dt.isocalendar().week.astype(int)
    data['WeekOfMonth'] = ((data['Day'].dt.day - 1) // 7 + 1).astype(int)

    cols = [
        'Year', 'Month', 'WeekOfYear', 'WeekOfMonth', 'Day', 'Hour',
        'ActiveClientsDay', 'ActiveClientsMonth', 'Patients',
        'Holiday', 'AfterHoliday', 'SchoolHoliday'
    ]
    original = data[cols].copy()

    for c in ['Hour', 'DayOfWeek']:
        data[c] = data[c].astype('category')
    data = pd.get_dummies(data, columns=['Hour', 'DayOfWeek'], drop_first=True)

    data.rename(columns=lambda col: re.sub(r'[^0-9A-Za-z_]', '_', col), inplace=True)

    return data, original


def main():
    config = get_connection_config()
    query = get_query()
    conn = get_connection(config)
    df = execute_query(conn, query)
    conn.close()

    print("Données extraites de la base de données :" f" {df.shape[0]} lignes, {df.shape[1]} colonnes")

    proc, orig = data_preprocessing(df)

    print("\nAprès data_preprocessing:"
          f" {proc.shape[0]} lignes, {proc.shape[1]} colonnes"
          f"\nColonnes : {proc.columns.tolist()}")
    print(proc.dtypes)
    print(proc.head())

    return proc, orig

if __name__ == "__main__":
    main()
