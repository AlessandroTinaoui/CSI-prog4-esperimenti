# ExtraTrees hyperparams (uguali per tutti i client per coerenza)
ET_N_ESTIMATORS = 200
ET_MAX_DEPTH = None
ET_MIN_SAMPLES_SPLIT = 2
ET_MIN_SAMPLES_LEAF = 1
ET_MAX_FEATURES = 1.0  # oppure 1.0 / "sqrt" (dipende dal tuo dataset)
ET_BOOTSTRAP = False


# ExtraTrees params (chiavi IDENTICHE a ExtraTreesRegressor)
EXTRA_TREES_PARAMS = {
    "n_estimators": ET_N_ESTIMATORS,
    "max_depth": ET_MAX_DEPTH,
    "max_features": ET_MAX_FEATURES,
    "bootstrap": ET_BOOTSTRAP,
    "min_samples_split": ET_MIN_SAMPLES_SPLIT,
    "min_samples_leaf": ET_MIN_SAMPLES_LEAF,
}
