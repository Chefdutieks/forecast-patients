# -*- coding: utf-8 -*-
"""
Created on Thu May 30 09:55:35 2024

@author: ALICE

Ce fichier construit le modèle de prévision du nombre de patients à partir des données historiques en utilisant un modèle de régression Random Forest. 
Il inclut des étapes de prétraitement des données, de formation et de test du modèle, et d'évaluation des performances du modèle.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import holidays
from vacances_scolaires_france import SchoolHolidayDates
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import get_train_data  

def data_preprocessing(data):
    """
    Préprocess les données en ajoutant des colonnes pertinentes et en effectuant un encodage.

    Args:
        data (DataFrame): Données brutes à prétraiter.

    Returns:
        DataFrame: Données prétraitées avec les colonnes d'origine pour référence.
    """
    data['Day'] = pd.to_datetime(data['Day'])
    
    # Déterminer les jours fériés
    all_holidays = holidays.France(years=range(2020, 2031))
    holiday_dates = set(pd.to_datetime(list(all_holidays.keys())))
    
    # Jour de grève des pharmacies qui a eu un impact important sur les patients (négatif) donc considéré comme un jour férié pour le modèle
    holiday_dates.add(pd.to_datetime('2024-05-30'))
    
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
    
    original_columns = data[['Year', 'WeekOfYear', 'Day', 'Hour', 'Minute', 'Holiday', 'AfterHoliday', 'ActiveClients']].copy()
    
    # Convertir les colonnes en catégories
    data['Hour'] = data['Hour'].astype('category')
    data['Minute'] = data['Minute'].astype('category')
    data['DayOfWeek'] = data['DayOfWeek'].astype('category')
    data['DayOfMonth'] = data['DayOfMonth'].astype('category')
    data['Month'] = data['Month'].astype('category')
    data['Year'] = data['Year'].astype('category')
    
    # Encodage One-Hot
    data = pd.get_dummies(data, columns=['Hour', 'Minute', 'DayOfWeek', 'DayOfMonth', 'Month', 'Year'], drop_first=True)
    
    # Convertir les colonnes booléennes en entiers
    bool_columns = data.select_dtypes(include='bool').columns
    data[bool_columns] = data[bool_columns].astype(int)
    
    return data, original_columns

def truncate_dataset(data, end_date):
    """
    Tronque le jeu de données jusqu'à une date spécifique.

    Args:
        data (DataFrame): Jeu de données à tronquer.
        end_date (str): Date de fin pour la troncature.

    Returns:
        DataFrame: Jeu de données tronqué.
    """
    end_date = pd.to_datetime(end_date)
    truncated_data = data[data['Day'] <= end_date]
    return truncated_data

def train_test_split_by_date(data, split_date):
    """
    Sépare les données en ensembles d'entraînement et de test basés sur une date de division.

    Args:
        data (DataFrame): Jeu de données à séparer.
        split_date (str): Date de division pour séparer les ensembles.

    Returns:
        tuple: Ensemble d'entraînement et de test pour les caractéristiques et les cibles.
    """
    train_data = data[data['Day'] <= split_date]
    test_data = data[data['Day'] > split_date] 
    X_train = train_data.drop(columns=['Day', 'Patients'])
    y_train = train_data['Patients']
    X_test = test_data.drop(columns=['Day', 'Patients'])
    y_test = test_data['Patients']
    return X_train, y_train, X_test, y_test

def plot_feature_importances(importances, feature_names):
    """
    Trace les importances des caractéristiques.

    Args:
        importances (array): Importances des caractéristiques.
        feature_names (list): Noms des caractéristiques.
    """
    indices = np.argsort(importances)
    plt.figure(figsize=(15, 15))
    plt.title('Importances des caractéristiques')
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importance relative')
    plt.show()

def random_forest(X_train, y_train, X_test, y_test, original_columns, split_date):
    """
    Entraîne et évalue un modèle Random Forest.

    Args:
        X_train (DataFrame): Caractéristiques d'entraînement.
        y_train (Series): Cibles d'entraînement.
        X_test (DataFrame): Caractéristiques de test.
        y_test (Series): Cibles de test.
        original_columns (DataFrame): Colonnes originales pour la comparaison.
        split_date (str): Date de division pour la comparaison.

    Returns:
        RandomForestRegressor: Modèle Random Forest entraîné.
    """
    # Création, entrainement et test du modèle
    rf_model = RandomForestRegressor(n_estimators=500, random_state=45)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    
    # Affichage des indicateurs 
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    print(f'RMSE (Random Forest): {rmse_rf}')
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    print(f'MAE (Random Forest): {mae_rf}')
    bias_rf = np.mean(y_pred_rf - y_test)
    print(f'Biais (Random Forest): {bias_rf}')
    variance_rf = np.var(y_pred_rf)
    print(f'Variance (Random Forest): {variance_rf}')
    
    # Afficher les importances des caractéristiques
    feature_importances = rf_model.feature_importances_
    feature_names = X_train.columns
    plot_feature_importances(feature_importances, feature_names)
    
    # Comparaisons valeurs réelles et prédites
    comparison_rf_graph = pd.DataFrame(X_test)
    comparison_rf_graph['Actual'] = y_test
    comparison_rf_graph['Predicted'] = y_pred_rf
    comparison_rf_csv = original_columns[original_columns['Day'] > split_date].copy()
    comparison_rf_csv['PredictedPatients'] = np.round(y_pred_rf).astype(int)
    comparison_rf_csv['ActualPatients'] = np.round(y_test.values).astype(int)
    #comparison_rf_csv.to_csv('detailed_predictions.csv', index=False)
    comparison_rf_sorted = comparison_rf_graph.sort_index()
    plt.figure(figsize=(12, 6))
    plt.plot(comparison_rf_sorted.index, comparison_rf_sorted['Actual'], label='Valeurs Réelles', color='blue')
    plt.plot(comparison_rf_sorted.index, comparison_rf_sorted['Predicted'], label='Valeurs Prédites', color='red', alpha=0.7)
    plt.title('Comparaison des Valeurs Réelles et Prédites avec Random Forest (Triées par Date)')
    plt.xlabel('Index')
    plt.ylabel('Nombre de Patients')
    plt.legend()
    plt.show()
    return rf_model

def main():
    """
    Fonction principale qui effectue la prévision des patients en utilisant un modèle Random Forest.

    Returns:
        RandomForestRegressor: Modèle Random Forest entraîné.
    """
    # Obtenir les données depuis get_train_data.main()
    dataset = get_train_data.main()
    
    # Prétraitement des données
    data, original_columns = data_preprocessing(dataset)
    
    # Déterminer les dates dynamiques
    last_date = data['Day'].max()
    end_date = last_date - pd.Timedelta(days=1)
    split_date = last_date - pd.Timedelta(days=2)
    
    # Troncature du dataset
    data = truncate_dataset(data, end_date)
    original_columns = truncate_dataset(original_columns, end_date)
    
    # Séparation en ensembles d'entraînement et de test
    X_train, y_train, X_test, y_test = train_test_split_by_date(data, split_date)
    
    # Entraînement et évaluation du modèle Random Forest
    rf_model = random_forest(X_train, y_train, X_test, y_test, original_columns, split_date)
    
    return rf_model

