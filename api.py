# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 15:36:56 2024

@author: alice

Ce fichier définit une API Flask pour prédire le nombre de patients en fonction de dates fournies. 
Il vérifie les autorisations via un token, valide les paramètres, charge un modèle de prédiction et renvoie les prédictions en format JSON.
L'API fonctionne pour le moment en local et devra probablement être adaptée lors du passage sur un serveur.
"""

# api.py
from flask import Flask, request, jsonify
import data_processing
import pickle
from datetime import datetime
import pandas as pd

from dotenv import load_dotenv
import os
load_dotenv() 

app = Flask(__name__)
VALID_TOKEN = os.getenv('API_VALID_TOKEN')

def validate_date(date_str):
    """
    Valide si une chaîne de caractères est une date valide au format YYYY-MM-DD.

    Args:
        date_str (str): Chaîne de caractères représentant une date.

    Returns:
        bool: True si la date est valide, False sinon.
    """
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False

def validate_spe(spe_str):
    """
    Valide si une chaîne de caractères peut être convertie en un entier dans une plage spécifique, correspondant aux spécialités possibles.

    Args:
        spe_str (str): Chaîne de caractères représentant un entier.

    Returns:
        bool: True si l'entier est dans la plage spécifiée, False sinon.
    """
    try:
        spe = int(spe_str)
        if 0 <= spe <= 38:
            return True
        return False
    except (ValueError, TypeError):
        return False

@app.before_request
def verify_token():
    """
    Vérifie le token d'autorisation avant chaque requête.
    """
    token = request.headers.get('Authorization')
    if token != VALID_TOKEN:
        return jsonify({'error': 'Unauthorized'}), 401

@app.route('/predict', methods=['GET'])
def predict():
    """
    Route de l'API pour prédire le nombre de patients sur une plage de dates donnée.

    Returns:
        Response: Réponse JSON contenant les prédictions.
    """
    start_date_str = request.args.get('start_date')
    end_date_str = request.args.get('end_date')
    spe = request.args.get('spe')

    if not start_date_str or not end_date_str:
        return jsonify({'error': 'Please provide start_date and end_date parameters'}), 400

    if not validate_date(start_date_str) or not validate_date(end_date_str):
        return jsonify({'error': 'Dates must be in the format YYYY-MM-DD'}), 400

    if spe is not None and not validate_spe(spe):
        return jsonify({'error': 'Parameter "spe" must be an integer between 0 and 38'}), 400

    # Charger le modèle de forêt aléatoire
    with open('rf_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    
    # Lire les données des feuilles Google Sheets
    existing_df_active_clients = data_processing.read_google_sheet("ActiveClients", "A1:E100")
    
    # Traiter les données des clients actifs
    active_clients_per_month = data_processing.process_active_clients(existing_df_active_clients)
    
    # Générer et prédire les données futures
    future_data = data_processing.generate_and_predict(start_date_str, end_date_str, active_clients_per_month, rf_model)

    # Convertir les colonnes Hour et Minute en entiers
    future_data['Hour'] = future_data['Hour'].astype(int)
    future_data['Minute'] = future_data['Minute'].astype(int)

    # Reformater les données pour ne garder que Datetime et PredictedPatients
    future_data['Datetime'] = pd.to_datetime(future_data['Day'], format='%d/%m/%Y') + pd.to_timedelta(future_data['Hour'], unit='h') + pd.to_timedelta(future_data['Minute'], unit='m')
    future_data['Datetime'] = future_data['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    future_data = future_data[['Datetime', 'PredictedPatients']]

    # Convertir les données futures en JSON
    future_data_json = future_data.to_dict(orient='records')

    return jsonify(future_data_json)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
