# -*- coding: utf-8 -*-
"""
Modified on Fri June 13 14:34:13 2025

@author: larrybird

Ce fichier a pour but de récupérer des données d'arrivée de patients depuis la base de données prod-kpi, et d'effectuer un traitement des valeurs manquantes sur ces données pour générer un jeu de données d'entraînement utilisable pour un modèle de prédiction. 
Le jeu de données contient des informations sur les patients et les clients actifs à différents créneaux horaires au cours de la journée. 
Changer de requête SQL pour inclure clients actifs par jour et par mois.
"""

import pymssql
import pandas as pd
import pyodbc
from dotenv import load_dotenv
import os
load_dotenv()


def get_connection_config():
    """
    Retourne la configuration de connexion.
    """
    config = {
        'server':   os.getenv('AZURE_DB_SERVER'),
        'database': os.getenv('AZURE_DB_NAME'),
        'user':     os.getenv('AZURE_DB_USER'),
        'password': os.getenv('AZURE_DB_PASSWORD'),
        'port':     1433
    }
    return config


def get_query():
    """
    Requête SQL pour extraire les données nécessaires avec clients actifs jour/mois.
    """
    query = """
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
    return query


def get_connection(config):
    """
    Connexion via pymssql pour Azure SQL Database.
    """
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
    """
    Exécute la requête SQL et retourne un DataFrame.
    """
    return pd.read_sql(query, connection)


def fill_missing(df):
    """
    Complète les créneaux horaires manquants (08:00 à 19:30, toutes les 30 minutes) pour chaque jour.
    """
    df['Day'] = pd.to_datetime(df['Day'])
    df['Hour'] = pd.to_datetime(df['Hour'], format='%H:%M').dt.time

    all_days = pd.date_range(start=df['Day'].min(), end=df['Day'].max(), freq='D')
    all_times = pd.date_range('08:00', '19:30', freq='30min').time

    all_slots = pd.MultiIndex.from_product([all_days, all_times], names=['Day', 'Hour']).to_frame(index=False)
    df_full = pd.merge(all_slots, df, on=['Day', 'Hour'], how='left')

    df_full['Patients'] = df_full['Patients'].fillna(0)
    df_full['ActiveClientsDay'] = df_full['ActiveClientsDay'].ffill().bfill()
    df_full['ActiveClientsMonth'] = df_full['ActiveClientsMonth'].ffill().bfill()

    return df_full


def main():
    """
    Fonction principale exécutant la chaîne complète : extraction SQL, traitement, et retour du DataFrame final.
    """
    config = get_connection_config()
    query = get_query()
    conn = get_connection(config)
    df = execute_query(conn, query)
    conn.close()

    # Appliquer le traitement des créneaux manquants
    df_processed = fill_missing(df)

    # Aperçu pour debug
    print("\nAprès traitement avec fill_missing:")
    print(df_processed.dtypes)
    print(df_processed.head())

    return df_processed


if __name__ == "__main__":
    main()
