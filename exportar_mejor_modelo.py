import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


RUTA_DATOS = "HOMICIDIO_Poblacion_Educacion_InnerJoin.xlsx"
RUTA_MODELO = "mejor_modelo_homicidios.joblib"
TARGET = "tasa_homicidios_100k"
COLUMNAS_EDUCACION = [
    "TECNICA PROFESIONAL",
    "TECNOLOGICA",
    "UNIVERSITARIA",
    "ESPECIALIZACION",
    "MAESTRIA",
    "DOCTORADO",
]


def preparar_datos():
    raw = pd.read_excel(RUTA_DATOS)
    df = raw.rename(
        columns={
            "AÑO": "año",
            "DEPARTAMENTO": "departamento",
            "CANTIDAD_HOMICIDIOS": "homicidios",
            "Población": "poblacion",
        }
    )
    df["departamento"] = df["departamento"].astype(str).str.strip().str.upper()
    df["año"] = pd.to_numeric(df["año"], errors="coerce").astype("Int64")
    df["homicidios"] = pd.to_numeric(df["homicidios"], errors="coerce")
    df["poblacion"] = pd.to_numeric(df["poblacion"], errors="coerce")
    df[TARGET] = (df["homicidios"] / df["poblacion"]) * 100000
    df = df.sort_values(["departamento", "año"]).reset_index(drop=True)

    grupo_dep = df.groupby("departamento", group_keys=False)
    df["tasa_lag_1"] = grupo_dep[TARGET].shift(1)
    df["tasa_lag_2"] = grupo_dep[TARGET].shift(2)
    df["homicidios_lag_1"] = grupo_dep["homicidios"].shift(1)
    df["tasa_media_3_anios"] = grupo_dep[TARGET].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )
    df["tasa_media_5_anios"] = grupo_dep[TARGET].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    df["tasa_std_3_anios"] = grupo_dep[TARGET].transform(
        lambda s: s.shift(1).rolling(3, min_periods=2).std()
    )
    df["cambio_tasa_1_anio"] = grupo_dep[TARGET].diff().shift(1)

    df["anio_centrado"] = df["año"] - df["año"].min()
    df["log_poblacion"] = np.log1p(df["poblacion"])
    df["educacion_total"] = df[COLUMNAS_EDUCACION].sum(axis=1)
    df["educacion_total_por_100k"] = (df["educacion_total"] / df["poblacion"]) * 100000
    df["universitaria_por_100k"] = (df["UNIVERSITARIA"] / df["poblacion"]) * 100000
    df["posgrado_por_100k"] = (
        (df["ESPECIALIZACION"] + df["MAESTRIA"] + df["DOCTORADO"]) / df["poblacion"]
    ) * 100000

    features = [
        "departamento",
        "anio_centrado",
        "log_poblacion",
        "tasa_lag_1",
        "tasa_lag_2",
        "homicidios_lag_1",
        "tasa_media_3_anios",
        "tasa_media_5_anios",
        "tasa_std_3_anios",
        "cambio_tasa_1_anio",
        "educacion_total_por_100k",
        "universitaria_por_100k",
        "posgrado_por_100k",
        *COLUMNAS_EDUCACION,
    ]
    df_modelo = df.dropna(subset=["tasa_lag_1", TARGET]).copy()
    return df_modelo, features


def crear_pipeline(features):
    categorical_features = ["departamento"]
    numeric_features = [col for col in features if col not in categorical_features]
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
            ("num", SimpleImputer(strategy="median"), numeric_features),
        ]
    )
    return Pipeline(
        [
            ("prep", preprocessor),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=400,
                    min_samples_leaf=2,
                    max_features=0.8,
                    random_state=42,
                ),
            ),
        ]
    )


def validar_alpha(df_modelo, features):
    alphas = np.linspace(0, 1, 21)
    filas_validacion = []

    for anio_validacion in [2017, 2018, 2019, 2020]:
        train_val = df_modelo[df_modelo["año"] < anio_validacion]
        validacion = df_modelo[df_modelo["año"] == anio_validacion]
        modelo_val = crear_pipeline(features)
        modelo_val.fit(train_val[features], train_val[TARGET])

        pred_ml = np.maximum(modelo_val.predict(validacion[features]), 0)
        pred_lag = validacion["tasa_lag_1"].values

        for alpha in alphas:
            pred_blend = (alpha * pred_lag) + ((1 - alpha) * pred_ml)
            filas_validacion.append(
                {
                    "alpha_lag": alpha,
                    "MAPE_validacion": mean_absolute_percentage_error(
                        validacion[TARGET], pred_blend
                    )
                    * 100,
                }
            )

    validacion_alpha = (
        pd.DataFrame(filas_validacion)
        .groupby("alpha_lag", as_index=False)["MAPE_validacion"]
        .mean()
        .sort_values("MAPE_validacion")
        .reset_index(drop=True)
    )
    return float(validacion_alpha.loc[0, "alpha_lag"]), validacion_alpha


df_modelo, features = preparar_datos()
alpha_lag, validacion_alpha = validar_alpha(df_modelo, features)

train = df_modelo[df_modelo["año"] < 2021].copy()
test = df_modelo[df_modelo["año"] == 2021].copy()

modelo_extra_trees = crear_pipeline(features)
modelo_extra_trees.fit(train[features], train[TARGET])

pred_ml = np.maximum(modelo_extra_trees.predict(test[features]), 0)
pred_final = (alpha_lag * test["tasa_lag_1"].values) + ((1 - alpha_lag) * pred_ml)

metricas = {
    "MAPE (%)": mean_absolute_percentage_error(test[TARGET], pred_final) * 100,
    "MAE": mean_absolute_error(test[TARGET], pred_final),
    "RMSE": np.sqrt(mean_squared_error(test[TARGET], pred_final)),
    "R2": r2_score(test[TARGET], pred_final),
}

artefacto = {
    "nombre_modelo": f"Mezcla Extra Trees + lag1 (alpha_lag={alpha_lag:.2f})",
    "modelo_extra_trees": modelo_extra_trees,
    "alpha_lag": alpha_lag,
    "features": features,
    "target": TARGET,
    "columnas_educacion": COLUMNAS_EDUCACION,
    "metricas_test_2021": metricas,
    "validacion_alpha": validacion_alpha,
    "instruccion_prediccion": (
        "pred_final = alpha_lag * X['tasa_lag_1'] + "
        "(1 - alpha_lag) * modelo_extra_trees.predict(X[features])"
    ),
}

joblib.dump(artefacto, RUTA_MODELO)

print("Modelo guardado en:", RUTA_MODELO)
print("Nombre:", artefacto["nombre_modelo"])
print("Metricas 2021:", metricas)
