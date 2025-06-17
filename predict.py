import os
import glob
import joblib
import pandas as pd
from datetime import datetime, timedelta
from TrainModels import evaluate_model
import get_train_data

def make_feature_matrix(df):
    # Supprime les colonnes non-utiles pour la prédiction
    return df.drop(columns=['Day', 'Patients'], errors='ignore')

def evaluate_models_on_historical(proc_df, orig_df, models_dir, eval_weeks=5):
    # Construction de X_all, y_all et dt_all
    X_all = make_feature_matrix(proc_df)
    y_all = proc_df['Patients'].values
    dt_all = pd.to_datetime(orig_df['Day'].astype(str)) + pd.to_timedelta(orig_df['Hour'].astype(str))

    # Période : dernières semaines complètes (lundi→dimanche)
    last_date    = dt_all.dt.date.max()
    last_sunday  = last_date - timedelta(days=(last_date.weekday() + 1) % 7)
    start_monday = last_sunday - timedelta(weeks=eval_weeks) + timedelta(days=1)

    mask   = (dt_all.dt.date >= start_monday) & (dt_all.dt.date <= last_sunday)
    X_eval = X_all.loc[mask].reset_index(drop=True)
    y_eval = y_all[mask]
    dt_eval= dt_all.loc[mask].reset_index(drop=True)

    preds_rows   = []
    metrics_rows = []

    for path in sorted(glob.glob(os.path.join(models_dir, 'best_*.pkl'))):
        name   = os.path.basename(path).replace('.pkl', '')
        model  = joblib.load(path)
        y_pred = model.predict(X_eval)

        # Collecte des prédictions
        for dt, actual, p in zip(dt_eval, y_eval, y_pred):
            preds_rows.append({
                'Datetime':          dt,
                'ActualPatients':    int(actual),
                'PredictedPatients': int(round(p)),
                'Model':             name
            })

        # Calcul des métriques
        m = evaluate_model(model, X_eval, y_eval)
        metrics_rows.append({
            'Model':    name,
            'RMSE':     m['RMSE'],
            'MAE':      m['MAE'],
            'Bias':     m['Bias'],
            'Variance': m['Variance']
        })

    df_preds   = pd.DataFrame(preds_rows)
    df_metrics = pd.DataFrame(metrics_rows)
    return df_preds, df_metrics

def pivot_predictions(df):
    # Transforme le format long en format large (une colonne Pred_<Modèle> par modèle)
    index_cols = ['Datetime']
    if 'ActualPatients' in df.columns:
        index_cols.append('ActualPatients')
    wide = df.pivot(index=index_cols,
                    columns='Model',
                    values='PredictedPatients')\
             .reset_index()
    wide.columns.name = None
    return wide

def generate_future_features(start_monday, weeks, proc_df):
    # Flags pour inclure ActiveClientsDay/Month
    flag_day   = os.getenv('ActiveClientsDay', 'True').lower() in ('true', '1', 'yes')
    flag_month = os.getenv('ActiveClientsMonth', 'True').lower() in ('true', '1', 'yes')

    # Période future complète (lundi→dimanche)
    end_sunday = start_monday + timedelta(weeks=weeks) - timedelta(days=1)

    # Créneaux de 08:30 à 19:30 tous les 30 min, hors dimanches
    slots = pd.date_range(
        start=start_monday + pd.Timedelta(hours=8, minutes=30),
        end=  end_sunday   + pd.Timedelta(hours=19, minutes=30),
        freq='30min'
    )
    slots = slots[
        ((slots.hour > 8) | ((slots.hour == 8) & (slots.minute >= 30))) &
        ((slots.hour < 19) | ((slots.hour == 19) & (slots.minute <= 30))) &
        (slots.dayofweek != 6)
    ]

    # Construire DataFrame pour prétraitement
    data = {
        'Day':      slots.date,
        'Hour':     slots.strftime('%H:%M'),
        'Patients': 1
    }
    # Imputation conditionnelle
    if flag_day:
        data['ActiveClientsDay'] = proc_df['ActiveClientsDay'].iloc[-1]
    if flag_month:
        data['ActiveClientsMonth'] = proc_df['ActiveClientsMonth'].iloc[-1]

    df_fut = pd.DataFrame(data)
    df_fut['Day'] = pd.to_datetime(df_fut['Day'])

    # Prétraitement existant
    proc_fut, _ = get_train_data.data_preprocessing(df_fut)
    X_fut = make_feature_matrix(proc_fut)
    return slots, X_fut

def forecast_models(proc_df, models_dir, forecast_weeks=13):
    # Calcul du prochain lundi après le dernier historique
    last_hist_day = proc_df['Day'].max().date()
    last_sunday   = last_hist_day - timedelta(days=(last_hist_day.weekday() + 1) % 7)
    next_monday   = last_sunday + timedelta(days=1)

    slots, X_fut = generate_future_features(next_monday, forecast_weeks, proc_df)
    rows = []
    for path in sorted(glob.glob(os.path.join(models_dir, 'best_*.pkl'))):
        name  = os.path.basename(path).replace('.pkl', '')
        model = joblib.load(path)
        y_pred = model.predict(X_fut)
        for dt, p in zip(slots, y_pred):
            rows.append({
                'Datetime':          dt,
                'PredictedPatients': int(round(p)),
                'Model':             name
            })
    return pd.DataFrame(rows)

def main():
    models_dir     = 'models/current'
    eval_weeks     = 5
    forecast_weeks = 13

    proc_df, orig_df = get_train_data.main()

    # Évaluation historique
    #df_preds, df_metrics = evaluate_models_on_historical(
    #    proc_df, orig_df, models_dir, eval_weeks
    #)
    ts = datetime.now().strftime('%Y%m%d_%H%M')

    #df_preds_wide = pivot_predictions(df_preds)
    #df_preds_wide.to_csv(f"historical_preds_{eval_weeks}w_{ts}.csv", index=False)
    #df_metrics.to_csv(f"historical_metrics_{eval_weeks}w_{ts}.csv", index=False)
    #print(f"Saved historical_preds_{eval_weeks}w_{ts}.csv")
    #print(f"Saved historical_metrics_{eval_weeks}w_{ts}.csv")

    # Prévision future
    df_forecast      = forecast_models(proc_df, models_dir, forecast_weeks)
    df_forecast_wide = pivot_predictions(df_forecast)
    df_forecast_wide.to_csv(f"forecast_preds_{forecast_weeks}w_{ts}.csv", index=False)
    print(f"Saved forecast_preds_{forecast_weeks}w_{ts}.csv")

if __name__ == '__main__':
    main()
