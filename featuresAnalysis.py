# -*- coding: utf-8 -*-
"""
feature_analysis.py - June 15, 2025

Module pour analyser la qualité des features issues de `get_train_data.main()` après
prétraitement via `data_preprocessing` :
- appel de la fonction `data_preprocessing` pour générer `processed_df`
- calcul et visualisation de la matrice de corrélation
- calcul du VIF (Variance Inflation Factor)
- suppression automatique des variables trop corrélées
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
import get_train_data
from TrainModels import data_preprocessing  # Assure-toi que models.py est dans le même répertoire


def plot_correlation_matrix(df, figsize=(12, 10), cmap='coolwarm'):
    corr = df.corr()
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=True, fmt='.2f', cmap=cmap, square=True, cbar_kws={'shrink': .8})
    plt.title('Matrice de corrélation des features')
    plt.tight_layout()
    plt.show()


def calculate_vif(df, features=None):
    X = df.copy()
    if features:
        X = X[features]
    else:
        X = X.select_dtypes(include=[np.number])
    X['__const__'] = 1
    vifs = []
    for i, col in enumerate(X.columns):
        if col == '__const__':
            continue
        vif = variance_inflation_factor(X.values, i)
        vifs.append({'feature': col, 'VIF': vif})
    return pd.DataFrame(vifs).sort_values('VIF', ascending=False)


def drop_highly_correlated(df, thresh=0.9):
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = [col for col in upper.columns if any(upper[col] > thresh)]
    reduced_df = df.drop(columns=drop_cols)
    return reduced_df, drop_cols


def feature_analysis_pipeline(df, target='Patients', corr_thresh=0.9, vif_thresh=10):
    # Exclure la colonne cible
    df_num = df.drop(columns=[target]) if target in df.columns else df.copy()
    df_num = df_num.select_dtypes(include=[np.number])

    print("\n=== 1. Matrice de corrélation ===")
    plot_correlation_matrix(df_num)

    print("\n=== 2. VIF initial ===")
    vif_initial = calculate_vif(df_num)
    print(vif_initial)

    print(f"\n=== 3. Suppression corrélation > {corr_thresh} ===")
    reduced_df, dropped = drop_highly_correlated(df_num, thresh=corr_thresh)
    print(f"Colonnes supprimées : {dropped}")

    print("\n=== 4. VIF après suppression ===")
    vif_post = calculate_vif(reduced_df)
    print(vif_post)

    high_vif = vif_post[vif_post['VIF'] > vif_thresh]
    if not high_vif.empty:
        print(f"\nAttention: variables avec VIF > {vif_thresh}:\n", high_vif)

    return reduced_df


if __name__ == '__main__':
    # Extraction et prétraitement des données
    df_raw = get_train_data.main()
    print("Données brutes extraites, shape=", df_raw.shape)

    # Appel à data_preprocessing
    processed_df, original_df = data_preprocessing(df_raw)
    print("Données prétraitées, shape=", processed_df.shape)
    # Aperçu des types de données
    print("\nAperçu des types de données après prétraitement:")
    print(processed_df.columns)

    # Analyse des features sur le DataFrame prétraité
    reduced_df = feature_analysis_pipeline(processed_df, target='Patients', corr_thresh=0.9, vif_thresh=10)
    print("Analyse terminée, shape réduite=", reduced_df.shape)
    # Optionnel : enregistrer reduced_df
    # reduced_df.to_csv('features_reduced.csv', index=False)
