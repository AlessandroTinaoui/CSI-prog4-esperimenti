from __future__ import annotations

import os
import glob
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer


# -----------------------------
# Config
# -----------------------------
@dataclass
class CleanConfig:
    # nomi colonne nel tuo CSV
    label_col: str = "label"      #nome colonna
    day_col: Optional[str] = "day"       # metti None se non vuoi considerarla speciale. nome della colonna che rappresenta un giorno

    # regole pulizia
    drop_label_zero: bool = True         #dopo aver tolto le label mancanti elimina anche le righe con label=0
    min_non_null_frac: float = 0.70      # "soglia minima per tenere una riga"
    iqr_k: float = 1.5                   # outlier IQR. controlla quanto lontano è un valore per essere considerato anomalo
    winsorize_iqr: bool = True           # True=valore schiacciato al limite, False=NaN (poi imputo)

    # imputazione
    knn_k: int =3                        #numero di righe vicine per stimare un NaN
    drop_all_nan_features: bool = True   # elimina feature rimaste NaN


import re

def drop_time_series_columns(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """
    Rimuove colonne time-series:
    - per nome (keyword)
    - per euristica: colonne object con stringhe molto lunghe (time series serializzate)
    """
    out = df.copy()
    name_keywords = ["time_series"]
    cols_by_name = [c for c in out.columns if any(k in c.lower() for k in name_keywords)]

    # euristica su colonne object: stringhe mediamente molto lunghe
    object_cols = [c for c in out.columns if out[c].dtype == "object"]
    cols_by_len = []
    for c in object_cols:
        s = out[c].dropna() #prende valori non Nan
        if s.empty: #se colonna tutta Nan salta
            continue
        s = s.astype(str) #converte tutto in stringa
        if s.map(len).mean() > 200:   # calcola la lunghezza di ogni cella, fa media lunghezze e se maggiore di 200 messa in cols_by_len
            cols_by_len.append(c)
    to_drop = sorted(set(cols_by_name + cols_by_len))#cols BY name:concatena set: elimina duplicati, sorted:ordine stabile
    if to_drop:
        out = out.drop(columns=to_drop, errors="ignore") #drop elimina

    return out


# -----------------------------
# Lettura CSV (pulita, senza magie)
# -----------------------------
def read_user_csv(path: str) -> pd.DataFrame:
    # I tuoi file sono tabelle separate da ';'
    df = pd.read_csv(path, sep=";")
    # spesso c'è una colonna indice inutile
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    return df


# -----------------------------
# Pulizia dataset singolo user
# -----------------------------

#funzione che ritorna la lista delle colonne feature numeriche da usare per la pulizia

def _select_numeric_feature_cols(df: pd.DataFrame, cfg: CleanConfig) -> List[str]:
    exclude = {cfg.label_col, "client_id", "user_id", "source_file"} #set colonne da escludere dalle feature
    if cfg.day_col:
        exclude.add(cfg.day_col) #se day non èNone viene escluso dalle feature

    cols = []
    for c in df.columns: #scorro tutto Data frame
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c) #se colonna numerica la aggiunge a cols
    return cols #ritorna elenco feature numeriche-lista di colonne numeriche utilizzabili

#prova a convertire le colonne non escluse in numeriche
def _coerce_numeric_features(df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    out = df.copy()
    exclude = {cfg.label_col, "client_id", "user_id", "source_file"}
    if cfg.day_col:
        exclude.add(cfg.day_col)

    for c in out.columns:
        if c in exclude:
            continue
        out[c] = pd.to_numeric(out[c], errors="ignore")#prova a convertire, se non riesce ignore
    return out #ritorna data frame con conversioni fatte

#elimina righe con label non valida
def drop_invalid_labels(df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    if cfg.label_col not in df.columns:
        raise ValueError(f"Colonna label '{cfg.label_col}' NON trovata. Colonne: {list(df.columns)}")

    out = df.dropna(subset=[cfg.label_col]).copy()
    if cfg.drop_label_zero:
        out = out[out[cfg.label_col] != 0]#se configurato elimina righe con label=0
    return out

#elimina righe troppo vuote
def drop_low_info_days(df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    out = df.copy()
    feat = _select_numeric_feature_cols(out, cfg)
    if not feat:
        return out

    frac = out[feat].notna().mean(axis=1)
    out = out[frac >= cfg.min_non_null_frac].copy() #ritorna solo le righe che hanno almeno un tot di dati (0,7 nel nostro caso)
    return out

#trova outlier con regola IQR e li tratta
def handle_outliers_iqr(df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    out = df.copy()
    feat = _select_numeric_feature_cols(out, cfg)

    for c in feat:
        s = out[c]
        # se pochi valori, evita di fare IQR
        if s.dropna().shape[0] < 5:
            continue

        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
#calcolo limiti superiore e inferiore
        lo = q1 - cfg.iqr_k * iqr
        hi = q3 + cfg.iqr_k * iqr

        if cfg.winsorize_iqr: #schiaccia tutto entro i limiti
            out[c] = s.clip(lo, hi)
        else: #gli outlier diventano NaN
            out.loc[(s < lo) | (s > hi), c] = np.nan

    return out

#Riempie i Nan delle feature numeriche con KNN imputer
def knn_impute(df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    out = df.copy()
    feat = _select_numeric_feature_cols(out, cfg)
    if not feat:
        return out#copia, prende feature numeriche e se non ce ne sono ritorna

    if cfg.drop_all_nan_features:
        all_nan = [c for c in feat if out[c].isna().all()]#trova le colonne tra le feature che sono tutte NaN
        if all_nan: #elimina NaN dal Data Frame
            out = out.drop(columns=all_nan)
            feat = [c for c in feat if c not in all_nan]
            if not feat:
                return out

    imputer = KNNImputer(n_neighbors=cfg.knn_k, weights="distance")
    out[feat] = imputer.fit_transform(out[feat].astype(float))#applica le imputazioni, calcola e riempie i NaN
    return out

#Copia data frame
def clean_user_df(df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    out = df.copy()

    out = drop_time_series_columns(out, cfg)#elimina sospette time series

    # opzionale: day come int (se è già numerico ok; se è stringa tipo "1", converte)
    if cfg.day_col and cfg.day_col in out.columns:
        out[cfg.day_col] = pd.to_numeric(out[cfg.day_col], errors="ignore")

    out = _coerce_numeric_features(out, cfg)#rende numeriche le colonne
    out = drop_invalid_labels(out, cfg)#toglie righe con label Nan
    out = drop_low_info_days(out, cfg)#toglie righe troppo incomplete
    out = handle_outliers_iqr(out, cfg)#gestisce outlier per colonna numerica
    out = knn_impute(out, cfg)#imputa i NaN rimanenti

    # ordina per day se presente
    if cfg.day_col and cfg.day_col in out.columns:
        out = out.sort_values(cfg.day_col)

    return out.reset_index(drop=True)


def _coerce_numeric_features_no_label(df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    """
    Versione di coercizione numerica che NON presume che label esista.
    Converte in numerico tutte le colonne non escluse (id/day/source_file).
    """
    out = df.copy()
    exclude = {"client_id", "user_id", "source_file"}
    if cfg.day_col:
        exclude.add(cfg.day_col)
    # NOTA: non escludiamo cfg.label_col perché in x_train potrebbe non esserci

    for c in out.columns:
        if c in exclude:
            continue
        out[c] = pd.to_numeric(out[c], errors="ignore")
    return out


def _select_numeric_feature_cols_no_label(df: pd.DataFrame, cfg: CleanConfig) -> List[str]:
    """
    Seleziona feature numeriche come _select_numeric_feature_cols,
    ma non richiede label e non la usa come esclusione se non presente.
    """
    exclude = {"client_id", "user_id", "source_file"}
    if cfg.day_col:
        exclude.add(cfg.day_col)
    # se la label esiste, la escludo
    if cfg.label_col in df.columns:
        exclude.add(cfg.label_col)

    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def handle_outliers_iqr_no_label(df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    """
    Stessa logica di handle_outliers_iqr ma selezione feature robusta anche senza label.
    """
    out = df.copy()
    feat = _select_numeric_feature_cols_no_label(out, cfg)

    for c in feat:
        s = out[c]
        if s.dropna().shape[0] < 5:
            continue

        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue

        lo = q1 - cfg.iqr_k * iqr
        hi = q3 + cfg.iqr_k * iqr

        if cfg.winsorize_iqr:
            out[c] = s.clip(lo, hi)
        else:
            out.loc[(s < lo) | (s > hi), c] = np.nan

    return out


def knn_impute_no_label(df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    """
    Stessa logica di knn_impute ma selezione feature robusta anche senza label.
    """
    out = df.copy()
    feat = _select_numeric_feature_cols_no_label(out, cfg)
    if not feat:
        return out

    if cfg.drop_all_nan_features:
        all_nan = [c for c in feat if out[c].isna().all()]
        if all_nan:
            out = out.drop(columns=all_nan)
            feat = [c for c in feat if c not in all_nan]
            if not feat:
                return out

    imputer = KNNImputer(n_neighbors=cfg.knn_k, weights="distance")
    out[feat] = imputer.fit_transform(out[feat].astype(float))
    return out


def preprocess_x_df(df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    """
    Preprocessing per x_train:
    - UGUALE alla pipeline dei dataset
    - MA NON elimina righe (no drop_invalid_labels, no drop_low_info_days)

    Funziona sia con che senza label.
    """
    out = df.copy()

    out = drop_time_series_columns(out, cfg)

    if cfg.day_col and cfg.day_col in out.columns:
        out[cfg.day_col] = pd.to_numeric(out[cfg.day_col], errors="ignore")

    out = _coerce_numeric_features_no_label(out, cfg)

    # niente drop righe

    out = handle_outliers_iqr_no_label(out, cfg)
    out = knn_impute_no_label(out, cfg)

    if cfg.day_col and cfg.day_col in out.columns:
        out = out.sort_values(cfg.day_col)

    return out.reset_index(drop=True)


# -----------------------------
# Merge per client (groupX)
# -----------------------------
#Definisce una funzione che prende in input il percorso e restituisce una stringa
def parse_user_id(filename: str) -> str:
    base = os.path.basename(filename).replace(".csv", "")#prende solo il nome file senza cartelle
    parts = base.split("_")#divide la stringa usando _
    if "user" in parts:#controlla se c'è la parola user
        i = parts.index("user")
        if i + 1 < len(parts):
            return parts[i + 1]
    return base

#legge tutti i group, unisce e pulisce i CSV e ritorna la lista di tuple
def build_clients(TRAIN_BASE_DIR: str, TRAIN_OUT_DIR: str, cfg: CleanConfig) -> List[Tuple[str, str]]:
    """
    TRAIN_BASE_DIR contiene group0, group1, ...
    dentro ogni group: dataset_user_*_train.csv
    """
    os.makedirs(TRAIN_OUT_DIR, exist_ok=True)#crea la cartella TRAIN_OUT_DIR

    group_dirs = sorted([d for d in glob.glob(os.path.join(TRAIN_BASE_DIR, "group*")) if os.path.isdir(d)])
    if not group_dirs:
        raise FileNotFoundError(f"Nessuna cartella group* trovata in: {TRAIN_BASE_DIR}")

    saved: List[Tuple[str, str]] = []

    for gdir in group_dirs:#scorre tutte le cartelle group trovate
        client_id = os.path.basename(gdir)  # prende il nome della cartella e lo usa come client_id
        user_files = sorted(glob.glob(os.path.join(gdir, "*.csv")))#cerca tutti i csv dentro quella cartela group e li ordina
        if not user_files:
            print(f"[WARN] {client_id}: nessun csv trovato.")
            continue

        parts = []#lista che conterrà i data frame puliti di ogni utente
        for p in user_files:
            user_id = parse_user_id(p)#estrae l'id utente del filename

            df = read_user_csv(p) #legge il csv con la funzione
            df["client_id"] = client_id #aggiunta colonna client_id
            df["user_id"] = user_id #colonna user_id
            df["source_file"] = os.path.basename(p) #colonna con nome file originale

            df = clean_user_df(df, cfg) #applica la pipeline di pulizia
            parts.append(df) #aggiunge data frame pulito a parts

        merged = pd.concat(parts, ignore_index=True) #concatena dataframe in verticale

        # opzionale: ordine stabile per (day, user)
        if cfg.day_col and cfg.day_col in merged.columns:
            merged = merged.sort_values([cfg.day_col, "user_id"]).reset_index(drop=True)

        out_path = os.path.join(TRAIN_OUT_DIR, f"{client_id}_merged_clean.csv") #costruisce il percorso file
        merged.to_csv(out_path, index=False) #salva il csv senza colonna indice

        print(f"[OK] {client_id}: {len(user_files)} utenti -> {merged.shape[0]} righe, {merged.shape[1]} colonne | {out_path}")
        saved.append((client_id, out_path))

    return saved

#blocco di esecuzione
if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    TRAIN_BASE_DIR = os.path.join(SCRIPT_DIR, "../raw_dataset")  # <- vedi la tua struttura annidata
    TRAIN_OUT_DIR  = os.path.join(SCRIPT_DIR, "clients_dataset")

    cfg = CleanConfig(
        label_col="label",
        day_col="day",
        min_non_null_frac=0.70,
        knn_k=3,
        iqr_k=1.5,
        winsorize_iqr=False
    )

    build_clients(TRAIN_BASE_DIR, TRAIN_OUT_DIR, cfg)

    # -----------------------------
    # Preprocessing X_test.csv (senza eliminare righe)
    # -----------------------------
    X_TEST_PATH = os.path.join(SCRIPT_DIR, "../raw_dataset/x_test.csv")  # <- cambia path se diverso
    X_TEST_OUT  = os.path.join(SCRIPT_DIR, "x_test_clean.csv")

    if os.path.exists(X_TEST_PATH):
        xdf = read_user_csv(X_TEST_PATH)
        xdf_clean = preprocess_x_df(xdf, cfg)
        xdf_clean.to_csv(X_TEST_OUT, index=False)
        print(f"[OK] x_test: {xdf.shape[0]} righe -> {xdf_clean.shape[0]} righe, "
              f"{xdf_clean.shape[1]} colonne | {X_TEST_OUT}")
    else:
        print(f"[WARN] x_test.csv non trovato: {X_TEST_PATH}")

