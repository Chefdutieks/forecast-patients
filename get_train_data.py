# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 14:34:13 2024

@author: alice

Ce fichier a pour but de récupérer des données d'arrivée de patients depuis la base de données prod-kpi, et d'effectuer un traitement des valeurs manquantes sur ces données pour générer un jeu de données d'entraînement utilisable pour un modèle de prédiction. 
Le jeu de données contient des informations sur les patients et les clients actifs à différents créneaux horaires au cours de la journée. 

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

    Returns:
        dict: Configuration de connexion à la base de données avec le serveur, le port, la base de données, l'utilisateur et le mot de passe.
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
    Retourne la requête SQL récupérant les données historiques d'arrivées de patients.

    Returns:
        str: Requête SQL pour extraire les données nécessaires.
    """
    query = """
    WITH PatientUnique AS (
        SELECT         
            PatientMatchings.PatientId as PatientId,
            CONVERT(date, PatientMatchings.StartedAt) AS Day,
            MIN(PatientMatchings.StartedAt) AS MaxStartedAt
        FROM PatientMatchings
        JOIN Clients AS Clients ON PatientMatchings.ClientId = Clients.Id
        WHERE (DATEPART(hh, PatientMatchings.StartedAt) BETWEEN 8 AND 20)
            AND (Clients.Name LIKE '%BOR%' OR Clients.Name LIKE '%CAB%' OR Clients.Name Like '%CBS' OR Clients.Name LIKE '%MAL%' OR Clients.Name LIKE '%CON%')
        GROUP BY
            PatientMatchings.PatientId,
            CONVERT(date, PatientMatchings.StartedAt)
    ),
    ActiveClientsByMonth AS (
        SELECT 
            CONVERT(varchar(7), StartedAt, 120) AS Month, -- YYYY-MM format
            COUNT(DISTINCT ClientId) AS ActiveClients
        FROM 
            PatientMatchings
            JOIN Clients AS Clients ON PatientMatchings.ClientId = Clients.Id
        WHERE 
            (Clients.Name LIKE '%BOR%' OR Clients.Name LIKE '%CAB%' OR Clients.Name LIKE '%CBS%' OR Clients.Name LIKE '%MAL%' OR Clients.Name LIKE '%CON%')
        GROUP BY 
            CONVERT(varchar(7), StartedAt, 120)
    )
    SELECT
        pu.Day,
        DATEPART(hh, pu.MaxStartedAt) AS Hour,
        CASE
            WHEN DATEPART(mi, pu.MaxStartedAt) < 30 THEN 0
            WHEN DATEPART(mi, pu.MaxStartedAt) >= 30 THEN 30
        END AS Minute,
        COUNT(DISTINCT pu.PatientId) AS Patients,
        ac.ActiveClients
    FROM
        PatientUnique pu
    JOIN
        ActiveClientsByMonth ac
        ON CONVERT(varchar(7), pu.Day, 120) = ac.Month
    GROUP BY
        pu.Day,
        DATEPART(hh, pu.MaxStartedAt),
        CASE
            WHEN DATEPART(mi, pu.MaxStartedAt) < 30 THEN 0
            WHEN DATEPART(mi, pu.MaxStartedAt) >= 30 THEN 30
        END,
        ac.ActiveClients
    ORDER BY
        pu.Day, Hour, Minute;
    """
    return query

def get_connection(config):
    """
    Connexion via pymssql pour Azure SQL Database.
    """
    return pymssql.connect(
        server   = config['server'],       # e.g. 'tessan-data.database.windows.net'
        port     = config['port'],         # 1433 (int)
        user     = config['user'],
        password = config['password'],
        database = config['database'],
        login_timeout = 30,
        # optionnel : forcer une version TDS récente si nécessaire
        tds_version = '7.4'
    )


def execute_query(connection, query):
    """
    Exécute la requête SQL et retourne un DataFrame.

    Args:
        connection (object): Objet de connexion à la base de données.
        query (str): Requête SQL à exécuter.

    Returns:
        DataFrame: Résultats de la requête sous forme de DataFrame.
    """
    return pd.read_sql(query, connection)


def fill_missing(df):
    """
    Préprocess les données récupérées en rajoutant les valeurs manquantes.

    Args:
        df (DataFrame): DataFrame contenant les données initiales.

    Returns:
        DataFrame: DataFrame prétraité avec les créneaux horaires complétés et les valeurs manquantes traitées.
    """

    # Convertir la colonne 'Day' en datetime pour les opérations suivantes
    df['Day'] = pd.to_datetime(df['Day'])
    
    # Générer tous les créneaux horaires entre 08:00 et 19:30 pour chaque jour présent dans le fichier
    all_days = pd.date_range(start=df['Day'].min(), end=df['Day'].max(), freq='D')
    all_times = pd.date_range(start='08:00', end='19:30', freq='30min').time
    
    all_slots = pd.MultiIndex.from_product([all_days, all_times], names=['Day', 'Time'])
    all_slots = pd.DataFrame(index=all_slots).reset_index()
    
    # Séparer l'heure et la minute de la colonne 'Time'
    all_slots['Hour'] = all_slots['Time'].apply(lambda x: x.hour)
    all_slots['Minute'] = all_slots['Time'].apply(lambda x: x.minute)
    all_slots = all_slots.drop(columns=['Time'])
    
    # Fusionner le DataFrame original avec le DataFrame des créneaux horaires
    df = pd.merge(all_slots, df, on=['Day', 'Hour', 'Minute'], how='left')
    
    # Remplir les valeurs manquantes
    df['Patients'] = df['Patients'].fillna(0)
    df['ActiveClients'] = df['ActiveClients'].ffill().bfill()

    return df


def main():
    """
    Fonction principale qui retourne le DataFrame d'entrainement final.

    Returns:
        DataFrame: DataFrame final après récupération et prétraitement des données.
    """
    # Obtenir la configuration de connexion
    config = get_connection_config()
    
    # Obtenir la requête SQL
    query = get_query()
    
    # Connexion à la base de données
    conn = get_connection(config)
    
    # Exécution de la requête et récupération des résultats dans un DataFrame
    df = execute_query(conn, query)
    
    # Fermeture de la connexion
    conn.close()
    
    # Traitement des valeurs manquantes
    df = fill_missing(df)
    
    return df