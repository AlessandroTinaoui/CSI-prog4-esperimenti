import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor


def load_data(csv_path: str):
    print(f"[INFO] Carico il dataset da: {csv_path}")
    df = pd.read_csv(csv_path, sep=";")

    # Teniamo solo colonne numeriche (escludiamo le time series stringa)
    numeric_df = df.select_dtypes(exclude="object").copy()

    if "label" not in numeric_df.columns:
        raise ValueError("Nel dataset non è presente la colonna 'label' (sleep score).")

    y = numeric_df["label"]
    X = numeric_df.drop(columns=["label", "Unnamed: 0"], errors="ignore")

    # Rimuovo colonne completamente vuote (tipo act_activeTime)
    cols_before = X.shape[1]
    X = X.dropna(axis=1, how="all")
    cols_after = X.shape[1]
    if cols_after < cols_before:
        print(f"[INFO] Rimosse {cols_before - cols_after} feature completamente NaN.")

    print(f"[INFO] Dataset numerico: {X.shape[0]} righe, {X.shape[1]} feature.")
    return X, y


def split_train_val_test(X, y, test_ratio=0.15, val_ratio=0.15, random_state=42):
    """
    Restituisce:
      X_train, X_val, X_test, y_train, y_val, y_test, X_train_val, y_train_val
    con proporzioni circa 70/15/15.
    """
    # Train+Val vs Test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=test_ratio,
        random_state=random_state
    )

    # Train vs Val (sul blocco train_val)
    val_ratio_adjusted = val_ratio / (1.0 - test_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_ratio_adjusted,
        random_state=random_state
    )

    print("[INFO] Split dataset:")
    print(f"  Train: {X_train.shape[0]} campioni")
    print(f"  Val:   {X_val.shape[0]} campioni")
    print(f"  Test:  {X_test.shape[0]} campioni")

    return X_train, X_val, X_test, y_train, y_val, y_test, X_train_val, y_train_val


def train_xgb_and_select_features(X_train, y_train, X_val, y_val, top_k=12):
    """
    Addestra un XGBoost su TRAIN, valuta su VALIDATION,
    e seleziona le top_k feature più importanti.
    """
    xgb_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("xgb", XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            tree_method="hist"  # veloce e moderno
        ))
    ])

    print("\n[INFO] Addestro XGBoost su TRAIN per selezione feature...")
    xgb_pipe.fit(X_train, y_train)

    # Valutazione su validation
    y_val_pred = xgb_pipe.predict(X_val)
    mae_val = mean_absolute_error(y_val, y_val_pred)
    r2_val = r2_score(y_val, y_val_pred)
    print(f"[VAL] MAE (validation): {mae_val:.3f}")
    print(f"[VAL] R²  (validation): {r2_val:.3f}")

    # Feature importance dal modello XGBoost
    xgb_model = xgb_pipe.named_steps["xgb"]
    importances = xgb_model.feature_importances_
    feat_imp = sorted(
        zip(X_train.columns, importances),
        key=lambda x: x[1],
        reverse=True
    )

    print("\n[INFO] Top feature per importanza (XGBoost):")
    for name, imp in feat_imp[:top_k]:
        print(f"  {name:35s} {imp:.3f}")

    top_features = [name for name, _ in feat_imp[:top_k]]
    return top_features


def train_final_model_and_evaluate(X_train_val, y_train_val, X_test, y_test, top_features):
    """
    Allena il modello finale XGBoost su TRAIN+VAL usando solo le top_features,
    e valuta su TEST (MAE e R²).
    """
    X_train_val_top = X_train_val[top_features]
    X_test_top = X_test[top_features]

    final_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("xgb", XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            tree_method="hist"
        ))
    ])

    print("\n[INFO] Addestro modello FINALE XGBoost su TRAIN+VAL con le top feature...")
    final_pipe.fit(X_train_val_top, y_train_val)

    y_test_pred = final_pipe.predict(X_test_top)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    r2_test = r2_score(y_test, y_test_pred)

    print("\n=== RISULTATI SU TEST SET (XGBoost) ===")
    print(f"[TEST] MAE (più basso è meglio): {mae_test:.3f}")
    print(f"[TEST] R²:                     {r2_test:.3f}")

    return final_pipe, mae_test, r2_test


def main():
    parser = argparse.ArgumentParser(
        description="XGBoost Regressor per predire lo sleep score usando le feature più utili."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="dataset_user_all_train.csv",
        help="Percorso al file CSV del dataset (default: dataset_user_all_train.csv)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=12,
        help="Numero di feature più importanti da usare nel modello finale (default: 12)"
    )

    args = parser.parse_args()

    # 1) Carico dati
    X, y = load_data(args.csv_path)

    # 2) Split train/val/test
    X_train, X_val, X_test, y_train, y_val, y_test, X_train_val, y_train_val = split_train_val_test(
        X, y,
        test_ratio=0.15,
        val_ratio=0.15,
        random_state=42
    )

    # 3) XGBoost su TRAIN per selezionare le feature più utili
    top_features = train_xgb_and_select_features(
        X_train, y_train,
        X_val, y_val,
        top_k=args.top_k
    )

    print("\n[INFO] Feature usate nel modello finale:")
    for f in top_features:
        print("  -", f)

    # 4) Modello finale XGBoost su TRAIN+VAL e valutazione su TEST
    _, mae_test, r2_test = train_final_model_and_evaluate(
        X_train_val, y_train_val,
        X_test, y_test,
        top_features
    )

    # Se vuoi, qui puoi salvare il modello:
    # from joblib import dump
    # dump(_, "xgb_sleep_model.joblib")


if __name__ == "__main__":
    main()
