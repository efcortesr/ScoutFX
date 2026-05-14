"""
market_value.py — ML Agent (Guía 02)
XGBoost Regresión para predecir valor de mercado + SHAP + detección de chollos.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os

REGRESSION_FEATURES = [
    "xG_per90","xAG_per90","SCA_per90","PrgP_per90","PrgR_per90","PrgC_per90",
    "Press%","Tkl+Int_per90","Cmp%_Med","G+A_per90","Age","Min",
    "league_encoded","position_encoded","squad_rank",
]

def train_market_value_model(df):
    """Entrena XGBoost para predecir log(market_value). Returns model, df_model, features, metrics."""
    df_model = df.dropna(subset=["market_value_in_eur"]).copy()
    df_model["log_value"] = np.log1p(df_model["market_value_in_eur"])
    if "league" in df_model.columns:
        df_model["league_encoded"] = df_model["league"].astype("category").cat.codes
    if "position_group" in df_model.columns:
        df_model["position_encoded"] = df_model["position_group"].astype("category").cat.codes
    if "squad" in df_model.columns and "market_value_in_eur" in df_model.columns:
        df_model["squad_rank"] = df_model.groupby("squad")["market_value_in_eur"].transform("mean").rank(pct=True)
    avail = [f for f in REGRESSION_FEATURES if f in df_model.columns]
    X = df_model[avail].fillna(0)
    y = df_model["log_value"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5,
                              min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                              random_state=42, n_jobs=-1)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)
    y_pred_log = model.predict(X_test)
    y_pred_eur = np.expm1(y_pred_log)
    y_test_eur = np.expm1(y_test)
    rmse = np.sqrt(mean_squared_error(y_test_eur, y_pred_eur))
    mae = mean_absolute_error(y_test_eur, y_pred_eur)
    r2 = r2_score(y_test, y_pred_log)
    # Baseline comparison
    baseline_pred = np.full_like(y_test, y_train.mean())
    baseline_rmse = np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(baseline_pred)))
    baseline_r2 = r2_score(y_test, baseline_pred)
    print(f"\n📊 Métricas XGBoost (test): RMSE: €{rmse/1e6:.2f}M | MAE: €{mae/1e6:.2f}M | R²: {r2:.3f}")
    print(f"📊 Baseline (media):       RMSE: €{baseline_rmse/1e6:.2f}M | R²: {baseline_r2:.3f}")
    os.makedirs("models/checkpoints", exist_ok=True)
    joblib.dump(model, "models/checkpoints/xgboost_market_value.pkl")
    df_model["predicted_value_eur"] = np.expm1(model.predict(X))
    df_model["value_ratio"] = df_model["predicted_value_eur"] / df_model["market_value_in_eur"]
    return model, df_model, avail, {"RMSE_M": rmse/1e6, "MAE_M": mae/1e6, "R2": r2}

def compute_shap_values(model, X, df_model, features):
    """Calcula SHAP values y detecta chollos."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    os.makedirs("docs/figures", exist_ok=True)
    shap.summary_plot(shap_values, X, plot_type="bar", show=False, max_display=12)
    plt.tight_layout()
    plt.savefig("docs/figures/shap_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    shap_df = pd.DataFrame(shap_values, columns=features, index=df_model.index)
    os.makedirs("data/processed", exist_ok=True)
    shap_df.to_csv("data/processed/shap_values.csv")
    chollos = df_model[(df_model["value_ratio"]>2.0)&(df_model["market_value_in_eur"]<15_000_000)].sort_values("value_ratio", ascending=False)
    chollos.to_csv("data/processed/chollos_detectados.csv", index=False)
    print(f"\n💎 Chollos detectados: {len(chollos)}")
    if len(chollos)>0:
        print(chollos[["player","squad","position_group","market_value_in_eur","predicted_value_eur","value_ratio"]].head(10).to_string())
    return shap_values, chollos

def plot_scatter_chollos(df_model):
    """Scatter plot valor real vs predicho con categorización de chollos."""
    df_plot = df_model.copy()
    df_plot["color"] = df_plot["value_ratio"].apply(
        lambda r: "Chollo (predicho > 2x actual)" if r>2
        else ("Sobrevalorado (predicho < 0.5x actual)" if r<0.5 else "Precio Justo"))
    fig = px.scatter(df_plot, x=df_plot["market_value_in_eur"]/1e6, y=df_plot["predicted_value_eur"]/1e6,
                     color="color",
                     hover_data=["player","squad","position_group","xG_per90","xAG_per90","PrgP_per90"],
                     color_discrete_map={"Chollo (predicho > 2x actual)":"#27ae60","Precio Justo":"#3498db",
                                         "Sobrevalorado (predicho < 0.5x actual)":"#e74c3c"},
                     title="Valor Real vs. Predicho (M€)",
                     labels={"x":"Valor Transfermarkt (M€)","y":"Valor predicho (M€)"})
    max_val = max(df_plot["market_value_in_eur"].max(), df_plot["predicted_value_eur"].max())/1e6
    fig.add_trace(go.Scatter(x=[0,max_val],y=[0,max_val],mode="lines",
                             line=dict(color="gray",dash="dash",width=1),name="Precio justo (y=x)"))
    fig.update_layout(height=600, plot_bgcolor="white", paper_bgcolor="white")
    return fig

def save_valued_dataset(df_model):
    os.makedirs("data/processed", exist_ok=True)
    df_model.to_csv("data/processed/players_valued.csv", index=False)
    print(f"✅ Dataset valorado guardado: {len(df_model)} jugadores")

if __name__ == "__main__":
    df = pd.read_csv("data/processed/players_clustered.csv")
    model, df_model, feats, metrics = train_market_value_model(df)
    X = df_model[[f for f in REGRESSION_FEATURES if f in df_model.columns]].fillna(0)
    compute_shap_values(model, X, df_model, feats)
    save_valued_dataset(df_model)
