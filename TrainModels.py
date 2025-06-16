# -*- coding: utf-8 -*-
"""
Refactored models.py - June 17, 2025
@author: Larrybird aka Chefdutieks

Module modulaire pour entraîner, optimiser dynamiquement et sauvegarder plusieurs modèles de régression
(avec RandomForest, XGBoost, LightGBM), exclusion des dimanches/jours fériés,
validation croisée glissante, early stopping, parallélisation, et gestion d'archives.
"""

import os
import shutil
import glob
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import holidays
import re
from vacances_scolaires_france import SchoolHolidayDates
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import get_train_data
import joblib

# ----------------------- Prétraitement -----------------------
def data_preprocessing(data):
    # Dates et renommage anciennes colonnes
    data['Day'] = pd.to_datetime(data['Day'])
    if 'ActiveClientsDay' in data.columns:
        data.rename(columns={'ActiveClientsDay': 'ActiveClients'}, inplace=True)

    # Jours fériés
    all_holidays = holidays.country_holidays('FR', years=range(2020, 2031))
    holiday_dates = set(pd.to_datetime(list(all_holidays.keys())))
    holiday_dates.add(pd.to_datetime('2024-05-30'))
    data['Holiday'] = data['Day'].isin(holiday_dates).astype(int)

    # Après jour férié (shift 48*30min)
    data = data.sort_values(['Day','Hour'])
    data['AfterHoliday'] = data['Day'].shift(48).isin(holiday_dates).astype(int)

    # Vacances scolaires
    school_holidays = SchoolHolidayDates()
    data['SchoolHoliday'] = data['Day'].apply(lambda x: school_holidays.is_holiday(x.date())).astype(int)

    # Caractéristiques temporelles
    data['DayOfWeek']   = data['Day'].dt.dayofweek 
    data['Month']       = data['Day'].dt.month
    data['Year']        = data['Day'].dt.year
    data['WeekOfYear']  = data['Day'].dt.isocalendar().week.astype(int)

    # Regroupement par semaine du mois (1-5)
    data['WeekOfMonth'] = ((data['Day'].dt.day - 1) // 7 + 1).astype(int)

    # Construction du jeu final (avant encodage)
    cols = ['Year','Month','WeekOfYear','WeekOfMonth','Day','Hour','ActiveClients','ActiveClientsMonth',
            'Patients','Holiday','AfterHoliday','SchoolHoliday']
    original = data[cols].copy()

    # Encodage catégoriel : uniquement Hour et DayOfWeek
    cats = ['Hour','DayOfWeek']
    for c in cats:
        data[c] = data[c].astype('category')
    data = pd.get_dummies(data, columns=cats, drop_first=True)

    # ─── SANITIZE FEATURE NAMES ────────────────────────────────────────
    # Remplace tout caractère non alphanumérique ou underscore par '_'
    data.rename(
        columns=lambda col: re.sub(r'[^0-9A-Za-z_]', '_', col),
        inplace=True
    )

    return data, original

# ----------------------- Exclusion des dates -----------------------
def filter_exclude(data):
    return data[(data['Day'].dt.dayofweek != 6) & (data['Holiday'] == 0)]

# ----------------------- Évaluation -----------------------
def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    return {
        'RMSE': np.sqrt(mean_squared_error(y, y_pred)),
        'MAE': mean_absolute_error(y, y_pred),
        'Bias': np.mean(y_pred - y),
        'Variance': np.var(y_pred)
    }

# ----------------------- Feature Importances -----------------------
def plot_feature_importance(model, features):
    if hasattr(model, 'feature_importances_'):
        imp = model.feature_importances_
        idx = np.argsort(imp)
        plt.figure(figsize=(12, 12))
        plt.barh(range(len(idx)), imp[idx])
        plt.yticks(range(len(idx)), [features[i] for i in idx])
        plt.title('Feature Importances')
        plt.tight_layout()
        plt.show()

# ----------------------- Graphiques Réels vs Prédits -----------------------
def plot_recent_predictions(model, X, orig, name, weeks):
    """
    Trace la série temporelle réelle vs prédite pour les dernières `weeks` semaines,
    avec un marker à chaque point.
    """
    # 1) Prépare un DataFrame filtré comme dans filter_exclude
    df = orig.copy().reset_index(drop=True)
    start = pd.to_datetime('08:30').time()
    end = pd.to_datetime('19:30').time()
    df = df[
        (df['Day'].dt.dayofweek != 6)
        & (df['Patients'] != 0)
        & (df['Hour'] >= start)
        & (df['Hour'] <= end)
    ]
    # 2) Axe des dates/heures
    datetimes = pd.to_datetime(df['Day'].astype(str)) + pd.to_timedelta(df['Hour'].astype(str))
    # 3) Filtrer sur les dernières `weeks` semaines
    cutoff = datetimes.max() - pd.Timedelta(weeks=weeks)
    mask = datetimes >= cutoff
    x_plot = datetimes[mask]
    # 4) Séries réelles et prédîtes
    y_actual = df['Patients'].values[mask.values]
    y_pred   = model.predict(X)[mask.values]
    # 5) Tracé avec markers
    plt.figure(figsize=(15,5))
    plt.plot(x_plot, y_pred,    marker='o', linestyle='-', label='Prédiction')
    plt.scatter(x_plot, y_actual, color='black', s=30, label='Réalité')
    plt.xlabel('Date & Heure')
    plt.ylabel('Patients')
    plt.title(f'{name} Prédictions vs Réel (dernières {weeks} semaines)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ----------------------- Modèles & hyperparamètres -----------------------
def get_models():
    return {
        'RandomForest': {
            'model': RandomForestRegressor(random_state=42, n_jobs=-1),
            'params': {'n_estimators': [100, 300], 'max_depth': [5, 10, None], 'min_samples_split': [2, 5]}
         },
        'XGBoost': {
            'model': xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1),
            'params': {'n_estimators': [100,300, 500], 'max_depth': [3, 10], 'learning_rate': [0.01, 0.1]}
        }#,
        #'LightGBM': {
        #    'model': lgb.LGBMRegressor(random_state=42, n_jobs=-1),
        #    'params': {'n_estimators': [100, 300], 'max_depth': [3, 10], 'learning_rate': [0.01, 0.1]}
        #}
    }

# ----------------------- Entraînement CV + GridSearch -----------------------
def train_with_cv(X, y, model, param_grid):
    tscv = TimeSeriesSplit(n_splits=5)
    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_root_mean_squared_error',
        verbose=1,
        n_jobs=-1
    )
    gs.fit(X, y)
    return gs.best_estimator_, gs.best_params_

# ----------------------- Gestion des chemins -----------------------
def prepare_output_dirs(base='models'):
    current = os.path.join(base, 'current')
    archive = os.path.join(base, 'archive')
    os.makedirs(current, exist_ok=True)
    os.makedirs(archive, exist_ok=True)
    return current, archive

# ----------------------- Pipeline principal -----------------------
def main(save_models=True):
    proc, orig = get_train_data.main()
    data = proc

    X = data.drop(columns=['Day', 'Patients'], errors='ignore')
    y = data['Patients']

    current_dir, archive_dir = prepare_output_dirs()
    results = {}

    for name, info in get_models().items():
        print(f"\n=== Entraînement {name} ===")
        best_model, best_params = train_with_cv(X, y, info['model'], info['params'])

        if name == 'XGBoost':
            best_model = xgb.XGBRegressor(
                objective='reg:squarederror',
                random_state=42,
                n_jobs=-1,
                **best_params,
                early_stopping_rounds=50,
                eval_metric='rmse'
            )
            # On passe uniquement eval_set à fit, sans callbacks
            best_model.fit(
                X, y,
                eval_set=[(X, y)],
                verbose=True
            )
        elif name == 'LightGBM':
            best_model.fit(
                X, y,
                eval_set=[(X, y)],
                callbacks=[lgb.early_stopping(20), lgb.reset_parameter(learning_rate=lambda r: r * 0.9)]
            )

        metrics = evaluate_model(best_model, X, y)
        results[name] = {'model': best_model, 'params': best_params, 'metrics': metrics}
        print(f"Params optimisés ({name}): {best_params}")
        print(f"Metrics ({name}): {metrics}")
        plot_feature_importance(best_model, X.columns.tolist())
        plot_recent_predictions(best_model, X, orig, name, weeks=6)

        # Sauvegarde du modèle
        if save_models:
            # Génère un timestamp unique
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # 1) Archiver tous les modèles existants pour ce name (ils ont déjà un timestamp)
            pattern = os.path.join(current_dir, f"best_{name}_*.pkl")
            for old_file in glob.glob(pattern):
                shutil.move(old_file, os.path.join(archive_dir, os.path.basename(old_file)))
                print(f"Archivé ancien {name} en {os.path.basename(old_file)}")

            # 2) Sauvegarder le nouveau modèle avec timestamp dans current/
            new_filename = f"best_{name}_{timestamp}.pkl"
            new_path = os.path.join(current_dir, new_filename)
            joblib.dump(best_model, new_path)
            print(f"Sauvegardé nouveau {name} dans {new_path}")

    return results

if __name__ == '__main__':
    main()
