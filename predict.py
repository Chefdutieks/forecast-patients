import os
import glob
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

from TrainModels import data_preprocessing, filter_exclude, evaluate_model
import get_train_data

def plot_predictions(datetimes, y_actual, y_pred, name):
    plt.figure(figsize=(15, 5))
    plt.plot(datetimes, y_actual, label='Actual', linewidth=2)
    plt.plot(datetimes, y_pred,    label='Predicted', alpha=0.7)
    plt.xlabel('Datetime')
    plt.ylabel('Patients')
    plt.title(name)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def generate_future_features(orig_df, proc_df, last_dt, weeks=16):
    # 1) Prochain lundi après last_dt
    next_mon_offset = (7 - last_dt.normalize().weekday()) % 7
    start_date = last_dt.normalize() + pd.Timedelta(days=next_mon_offset)

    # 2) Générer datetimes tous les 30min
    periods = weeks * 7 * 48
    future_datetimes = pd.date_range(start=start_date, periods=periods, freq='30T')
    # Construire le masque pour filtrer :
    # - les heures entre 8h30 et 19h30
    # - exclure les dimanches (dayofweek == 6)
    allowed_mask = (
        ((future_datetimes.hour > 8) | ((future_datetimes.hour == 8) & (future_datetimes.minute >= 30))) &
        ((future_datetimes.hour < 19) | ((future_datetimes.hour == 19) & (future_datetimes.minute <= 30))) &
        (future_datetimes.dayofweek != 6)  # 6 correspond à dimanche
    )
    filtered_future_datetimes = future_datetimes[allowed_mask]

    # 3) Construire DataFrame avec heure au format identique au training
    future_df = pd.DataFrame({
        'Day': filtered_future_datetimes.date,
        'Hour': filtered_future_datetimes.strftime('%H:%M:%S')
    })
    future_df['Day'] = pd.to_datetime(future_df['Day'])

    # 4) Imputation ActiveClients et ActiveClientsMonth
    last_day = proc_df['Day'].max()
    last_ac = proc_df.loc[proc_df['Day'] == last_day, 'ActiveClients'].iloc[-1]
    last_acm = proc_df.loc[proc_df['Day'] == last_day, 'ActiveClientsMonth'].iloc[-1]
    future_df['ActiveClients'] = last_ac
    future_df['ActiveClientsMonth'] = last_acm

    # 5) Ajouter Patients dummy
    future_df['Patients'] = 0

    # 6) Prétraiter et filtrer comme historique
    proc_fut, _ = data_preprocessing(future_df)
    proc_fut = filter_exclude(proc_fut).reset_index(drop=True)

    # 7) Construire X_future
    X_future = proc_fut.drop(columns=['Day', 'Patients'], errors='ignore')
    return future_datetimes, X_future

def main(models_dir='models/current', eval_weeks=6, forecast_weeks=16):
    # Charger données
    df_raw = get_train_data.main()
    proc_df, orig_df = data_preprocessing(df_raw)
    proc_df = filter_exclude(proc_df).reset_index(drop=True)
    orig_df = filter_exclude(orig_df).reset_index(drop=True)

    # Construire X,y,datetimes
    X_all = proc_df.drop(columns=['Day', 'Patients'], errors='ignore')
    y_all = proc_df['Patients'].values
    datetimes = pd.to_datetime(orig_df['Day'].astype(str)) + pd.to_timedelta(orig_df['Hour'].astype(str))

    # Évaluation historique
    last_dt = datetimes.max()
    cutoff_eval = last_dt - pd.Timedelta(weeks=eval_weeks)
    # Remonter au lundi
    monday_off = cutoff_eval.weekday()
    start_eval = (cutoff_eval - pd.Timedelta(days=monday_off)).normalize()
    mask_eval = (datetimes >= start_eval) & (datetimes <= last_dt)
    X_eval = X_all[mask_eval]
    y_eval = y_all[mask_eval]
    dt_eval = datetimes[mask_eval]

    print(f"Evaluation sur {eval_weeks} dernières semaines : du {start_eval.date()} au {last_dt.date()}")

    # Prévision future
    future_dates, X_future = generate_future_features(orig_df, proc_df, last_dt, weeks=forecast_weeks)
    print(f"Prévision sur {forecast_weeks} prochaines semaines : du {future_dates.min().date()} au {future_dates.max().date()}")

    # Boucle sur modèles
    pkl_files = sorted(glob.glob(os.path.join(models_dir, 'best_*.pkl')))
    for path in pkl_files:
        name = os.path.basename(path).replace('.pkl', '')
        model = joblib.load(path)
        print(f"\n=== Modèle : {name} ===")

        # a) Évaluation historique
        y_pred_eval = model.predict(X_eval)
        metrics = evaluate_model(model, X_eval, y_eval)
        print(f"RMSE={metrics['RMSE']:.2f}, MAE={metrics['MAE']:.2f}, Bias={metrics['Bias']:.2f}, Variance={metrics['Variance']:.2f}")
        plot_predictions(dt_eval, y_eval, y_pred_eval, f"{name} - Évaluation")

        # b) Prévision future
        y_pred_fut = model.predict(X_future)
        plot_predictions(future_dates, y_pred_fut, y_pred_fut, f"{name} - Prévision {forecast_weeks} sem.")

if __name__ == '__main__':
    main()
