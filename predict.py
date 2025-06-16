import os
import glob
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from TrainModels import evaluate_model, get_feature_params
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

def generate_future_features(last_dt, weeks=13):
    next_mon_offset = (7 - last_dt.normalize().weekday()) % 7
    start_date = last_dt.normalize() + pd.Timedelta(days=next_mon_offset)

    periods = weeks * 7 * 48
    future_datetimes = pd.date_range(start=start_date, periods=periods, freq='30min')

    allowed_mask = (
        ((future_datetimes.hour > 8) | ((future_datetimes.hour == 8) & (future_datetimes.minute >= 30))) &
        ((future_datetimes.hour < 19) | ((future_datetimes.hour == 19) & (future_datetimes.minute <= 30))) &
        (future_datetimes.dayofweek != 6)
    )
    filtered_future_datetimes = future_datetimes[allowed_mask]

    future_df = pd.DataFrame({
        'Day': filtered_future_datetimes.date,
        'Hour': filtered_future_datetimes.strftime('%H:%M:%S')
    })
    future_df['Day'] = pd.to_datetime(future_df['Day'])

    proc_fut_av,orig = get_train_data.data_preprocessing(future_df)
    proc_fut=get_feature_params(proc_fut_av, active_clients_Day=False, active_clients_month=False)
    X_future = proc_fut.drop(columns=['Day', 'Patients'], errors='ignore')
    return filtered_future_datetimes, X_future

def main(models_dir='models/current', eval_weeks=5, forecast_weeks=13):
    proc_df_av, orig_df_av = get_train_data.main()
    proc_df = get_feature_params(proc_df_av, active_clients_Day=False, active_clients_month=False)
    orig_df = get_feature_params(orig_df_av, active_clients_Day=False, active_clients_month=False)
    X_all = proc_df.drop(columns=['Day', 'Patients'], errors='ignore')
    y_all = proc_df['Patients'].values
    datetimes = pd.to_datetime(orig_df['Day'].astype(str)) + pd.to_timedelta(orig_df['Hour'].astype(str))

    last_dt = datetimes.max()
    cutoff_eval = last_dt - pd.Timedelta(weeks=eval_weeks)
    monday_off = cutoff_eval.weekday()
    start_eval = (cutoff_eval - pd.Timedelta(days=monday_off)).normalize()
    mask_eval = (datetimes >= start_eval) & (datetimes <= last_dt)
    X_eval = X_all[mask_eval]
    y_eval = y_all[mask_eval]
    dt_eval = datetimes[mask_eval]

    print(f"Evaluation sur {eval_weeks} dernières semaines : du {start_eval.date()} au {last_dt.date()}")

    future_dates, X_future = generate_future_features(last_dt, weeks=forecast_weeks)
    print(f"Prévision sur {forecast_weeks} prochaines semaines : du {future_dates.min().date()} au {future_dates.max().date()}")

    all_forecasts = []

    for path in sorted(glob.glob(os.path.join(models_dir, 'best_*.pkl'))):
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

        # Collecte des données
        future_df = pd.DataFrame({
            'Datetime': future_dates,
            'PredictedPatients': np.round(y_pred_fut).astype(int),
            'Model': name
        })
        all_forecasts.append(future_df)

    # Timestamp pour le nom du fichier
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    final_filename = f"all_models_forecast_{forecast_weeks}weeks_{timestamp}.csv"

    # Sauvegarde finale
    full_df = pd.concat(all_forecasts)
    full_df.to_csv(final_filename, index=False)
    print(f"Fichier CSV global sauvegardé : {final_filename}")

if __name__ == '__main__':
    main()
