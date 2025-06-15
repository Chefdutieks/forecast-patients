import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def analyze_doctor_patient_ratio(csv_path):
    df = pd.read_csv(csv_path)
    df['Datetime'] = pd.to_datetime(df['Day'] + ' ' + df['Hour'], errors='coerce')
    df = df.dropna(subset=['waitingtime', 'Doctors', 'Patients', 'Datetime'])
    df = df[df['Patients'] > 0]
    df['R'] = df['Doctors'] / df['Patients']

    # Ratio idéal entre 10 et 15 minutes d’attente
    target = df[(df['waitingtime'] >= 10) & (df['waitingtime'] <= 15)].copy()
    ideal_ratio = target['R'].mean()
    print(f"🎯 Ratio idéal (10–15 min d’attente) : {ideal_ratio:.3f} → 1 médecin pour ~{1/ideal_ratio:.2f} patients")

    # Vérification : cas extrêmes
    over_20 = df[df['waitingtime'] > 20]
    under_5 = df[df['waitingtime'] < 5]
    tol = 0.02
    match_over = over_20[(over_20['R'] >= ideal_ratio - tol) & (over_20['R'] <= ideal_ratio + tol)]
    match_under = under_5[(under_5['R'] >= ideal_ratio - tol) & (under_5['R'] <= ideal_ratio + tol)]
    pct_over = len(match_over) / len(over_20) * 100 if len(over_20) > 0 else 0
    pct_under = len(match_under) / len(under_5) * 100 if len(under_5) > 0 else 0
    print(f"🔎 Ratio idéal avec attente >20min : {pct_over:.2f}%")
    print(f"🔎 Ratio idéal avec attente <5min : {pct_under:.2f}%")

    # Analyse par demi-heure (hors 20:00 et 20:30)
    target['Slot'] = target['Datetime'].dt.strftime('%H:%M')
    target = target[~target['Slot'].isin(['20:00', '20:30'])]
    slot_ratio = target.groupby('Slot')['R'].mean().reset_index()

    plt.figure(figsize=(12, 5))
    sns.barplot(data=slot_ratio, x='Slot', y='R', color='skyblue')
    plt.axhline(ideal_ratio, linestyle='--', color='red', label='Ratio idéal')
    plt.xticks(rotation=45)
    plt.title("Ratio idéal par demi-heure (hors 20h+)")
    plt.ylabel("Ratio médecins / patients")
    plt.xlabel("Créneau")
    plt.ylim(0, slot_ratio['R'].max() * 1.2)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Analyse par jour de la semaine (sans dimanche)
    target['Weekday'] = target['Datetime'].dt.day_name()
    weekday = target[target['Weekday'] != 'Sunday']
    weekday_ratio = weekday.groupby('Weekday')['R'].mean().reset_index()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    weekday_ratio['Weekday'] = pd.Categorical(weekday_ratio['Weekday'], categories=order, ordered=True)
    weekday_ratio = weekday_ratio.sort_values('Weekday')

    plt.figure(figsize=(10, 5))
    sns.barplot(data=weekday_ratio, x='Weekday', y='R', color='lightgreen')
    plt.axhline(ideal_ratio, linestyle='--', color='red', label='Ratio idéal')
    plt.title("Ratio idéal par jour de la semaine")
    plt.ylabel("Ratio médecins / patients")
    plt.xlabel("Jour")
    plt.ylim(0, weekday_ratio['R'].max() * 1.2)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Analyse lendemain de jour férié
    holidays = pd.to_datetime([
        '2022-07-14', '2022-11-01', '2022-12-25', '2023-01-01', '2023-05-01', '2023-07-14',
        '2023-08-15', '2023-11-01', '2023-12-25', '2024-01-01', '2024-05-01', '2024-07-14'
    ])
    df['Date'] = df['Datetime'].dt.normalize()
    df['IsAfterHoliday'] = df['Date'].isin(holidays + pd.Timedelta(days=1))
    after_holiday = df[df['IsAfterHoliday']]
    after_holiday_ratio = after_holiday[
        (after_holiday['waitingtime'] >= 10) & (after_holiday['waitingtime'] <= 15)
    ]['R'].mean()
    print(f"📆 Ratio moyen le lendemain d’un jour férié : {after_holiday_ratio:.3f}")

    return ideal_ratio, pct_under, pct_over, slot_ratio, weekday_ratio, after_holiday_ratio

# Exemple :
ideal, pct_under, pct_over, slots, weekdays, after_holiday = analyze_doctor_patient_ratio("Waitingtime,doctor,patients.csv")
print(f"Ratio idéal : {ideal:.3f}, % sous 5 min : {pct_under:.2f}%, % au-dessus de 20 min : {pct_over:.2f}%")
print(f"Ratio le lendemain d'un jour férié : {after_holiday:.3f}")
print("Ratios par créneau horaire :")
print(slots)
print("Ratios par jour de la semaine :")
print(weekdays)