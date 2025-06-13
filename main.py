# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 15:36:33 2024

@author: alice

Ce fichier principal planifie une tâche hebdomadaire (à ajuster si besoin) pour lire des données depuis Google Sheets, entraîner le modèle de forêt aléatoire, prédire des données futures, écrire ces prédictions et sauvegarder le modèle pour une utilisation future.
"""

# main.py

import random_forest_model
import matplotlib
import data_processing
import pickle
matplotlib.use('Agg')

def scheduled_task():
    """
    Tâche planifiée pour lire les données, entraîner le modèle, prédire les données futures et sauvegarder le modèle.
    """
    # Lire les données des feuilles Google Sheets
    existing_df_active_clients = data_processing.read_google_sheet("ActiveClients", "A1:E100")
    existing_df_google_api = data_processing.read_google_sheet("google api", "A1:G")
    
    # Traiter les données des clients actifs
    active_clients_per_month = data_processing.process_active_clients(existing_df_active_clients)
    
    # Choix de la période de prédiction
    start_date_str, end_date_str = data_processing.get_date_range(16)
    
    # Création et entrainement d'un nouveau modèle 
    rf_model = random_forest_model.main()
    print('model created')
    
    # Générer et prédire les données futures
    future_data = data_processing.generate_and_predict(start_date_str, end_date_str, active_clients_per_month, rf_model)
    print('data predict')
    # Écrire les données dans Google Sheets
    data_processing.write_sheet(future_data, existing_df_google_api)
    print('written on sheets')
    
    
    # Sauvegarder le modèle pour l'API
    with open('rf_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    print("Modèle sauvegardé")

# if __name__ == '__main__':
   #import schedule
    #import time
    #import pickle

    # Planifier la tâche d'entraînement du modèle toutes les semaines
    #schedule.every().day.at("13:37").do(scheduled_task)
    #schedule.every().sunday.at("00:00").do(scheduled_task)

    #while True:
        #schedule.run_pending()
        #time.sleep(1)

scheduled_task()

# Template du fichier qu'on veut importer 