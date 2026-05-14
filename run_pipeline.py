"""
run_pipeline.py — Ejecuta el pipeline completo de ScoutFX.
Paso 1: Carga Transfermarkt (FBref se salta por rate limits, usa datos sintéticos)
Paso 2: Preprocessing + Feature Engineering
Paso 3: Clustering (KMeans + UMAP)
Paso 4: XGBoost + SHAP + Chollos
Paso 5: Embeddings + ChromaDB
"""
import sys
import os

# Asegurar que el path incluya la raíz del proyecto
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

print("=" * 60)
print("  ScoutFX — Pipeline Completo")
print("=" * 60)

# ─── PASO 1: Cargar Transfermarkt ───
print("\n[1/5] Cargando datos de Transfermarkt...")
from src.data.loader_transfermarkt import load_transfermarkt

tm = load_transfermarkt("data/raw/players.csv")
print(f"  Columnas disponibles: {list(tm.columns[:10])}...")
print(f"  Total registros: {len(tm)}")

# ─── Verificar columnas del CSV ───
print(f"\n  Todas las columnas del CSV:")
for i, col in enumerate(tm.columns):
    print(f"    {i}: {col}")

# ─── PASO 1b: Preparar datos sin FBref (usar Transfermarkt como base) ───
print("\n[1b] Preparando dataset base desde Transfermarkt...")

# Crear features simuladas basadas en posición para el pipeline
# En producción, estas vendrían de FBref via soccerdata
np.random.seed(42)

# Mapear posiciones de Transfermarkt
pos_map = {
    "Attack": "FW", "attack": "FW",
    "Midfield": "MF", "midfield": "MF",
    "Defender": "CB", "defender": "CB",
    "Goalkeeper": "GK", "goalkeeper": "GK",
}

# Detectar columna de posición
pos_col = None
for c in ["position", "sub_position", "player_position"]:
    if c in tm.columns:
        pos_col = c
        break

if pos_col:
    tm["position_group"] = tm[pos_col].astype(str).apply(
        lambda x: next((v for k, v in pos_map.items() if k.lower() in x.lower()), "MF")
    )
    # Refinar: laterales
    if "sub_position" in tm.columns:
        mask_fb = tm["sub_position"].astype(str).str.lower().str.contains("back|full|lateral|wing-back", na=False)
        mask_df = tm["position_group"] == "CB"
        tm.loc[mask_fb & mask_df, "position_group"] = "FB"
        # Extremos / mediapuntas
        mask_am = tm["sub_position"].astype(str).str.lower().str.contains("wing|attacking|second striker", na=False)
        mask_fw = tm["position_group"] == "FW"
        tm.loc[mask_am, "position_group"] = "AM"
else:
    tm["position_group"] = "MF"

print(f"  Distribución por posición:")
print(tm["position_group"].value_counts().to_string())

# Crear features estadísticas basadas en valor de mercado + ruido
# Esto permite que el pipeline funcione end-to-end
# En producción, se reemplazaría con datos reales de FBref
n = len(tm)
log_value = np.log1p(tm["market_value_in_eur"].values)
percentile = (log_value - log_value.min()) / (log_value.max() - log_value.min() + 1e-9)

# Asignar nombre de jugador consistente
if "player_name" in tm.columns and "player" not in tm.columns:
    tm["player"] = tm["player_name"]

# Club
for col_name in ["current_club_name", "club_name", "squad"]:
    if col_name in tm.columns:
        tm["squad"] = tm[col_name]
        break

# Liga
for col_name in ["current_club_domestic_competition_id", "league", "competition_id"]:
    if col_name in tm.columns:
        tm["league"] = tm[col_name]
        break

# Edad
if "date_of_birth" in tm.columns and "Age" not in tm.columns:
    try:
        tm["Age"] = (pd.Timestamp("2024-06-01") - pd.to_datetime(tm["date_of_birth"])).dt.days / 365.25
    except:
        tm["Age"] = 25

# Generar stats sintéticas correlacionadas con el valor
tm["Min"] = np.random.randint(900, 3200, n)
tm["xG_per90"] = percentile * 0.6 + np.random.normal(0, 0.1, n)
tm["xAG_per90"] = percentile * 0.3 + np.random.normal(0, 0.08, n)
tm["PrgP_per90"] = percentile * 8 + np.random.normal(3, 1.5, n)
tm["PrgR_per90"] = percentile * 5 + np.random.normal(2, 1, n)
tm["PrgC_per90"] = percentile * 4 + np.random.normal(1.5, 0.8, n)
tm["Press%"] = percentile * 15 + np.random.normal(25, 5, n)
tm["Tkl+Int_per90"] = percentile * 2 + np.random.normal(2.5, 0.8, n)
tm["Cmp%_Med"] = percentile * 10 + np.random.normal(80, 5, n)
tm["SCA_per90"] = percentile * 3 + np.random.normal(2, 0.8, n)
tm["KP_per90"] = percentile * 2 + np.random.normal(1, 0.5, n)
tm["G+A_per90"] = percentile * 0.5 + np.random.normal(0.1, 0.1, n)
tm["Sh_per90"] = percentile * 3 + np.random.normal(1.5, 0.8, n)
tm["SoT%"] = percentile * 15 + np.random.normal(30, 8, n)
tm["G_per90"] = percentile * 0.4 + np.random.normal(0.05, 0.08, n)
tm["Succ%_Drib"] = percentile * 15 + np.random.normal(45, 10, n)
tm["Carries_1/3_per90"] = percentile * 3 + np.random.normal(1, 0.5, n)
tm["Carries_PrgDist_per90"] = percentile * 50 + np.random.normal(100, 30, n)
tm["AerWon%"] = np.random.normal(50, 12, n)
tm["Cmp%_Long"] = np.random.normal(55, 10, n)
tm["Blocks_per90"] = np.random.normal(1.5, 0.5, n)
tm["Tkl_per90"] = tm["Tkl+Int_per90"] * 0.6 + np.random.normal(0, 0.3, n)
tm["Int_per90"] = tm["Tkl+Int_per90"] * 0.4 + np.random.normal(0, 0.2, n)
tm["Clr_per90"] = np.random.normal(3, 1, n)
tm["CrsPA_per90"] = percentile * 1.5 + np.random.normal(0.5, 0.3, n)
tm["TklWon%"] = np.random.normal(60, 10, n)
tm["KP_per90"] = percentile * 2 + np.random.normal(1, 0.5, n)
tm["Att3rd_Tkl%"] = np.random.normal(30, 8, n)
tm["PrgDist_per90"] = percentile * 100 + np.random.normal(200, 50, n)
tm["Pos"] = tm["position_group"]

# Clip valores negativos
for col in tm.select_dtypes(include=[np.number]).columns:
    if col not in ["Age", "market_value_in_eur", "player_id"]:
        tm[col] = tm[col].clip(lower=0)

# Filtrar solo jugadores con valor de mercado
tm = tm[tm["market_value_in_eur"] > 0].copy()

# Guardar como dataset "final" para el pipeline
os.makedirs("data/processed", exist_ok=True)
tm.to_csv("data/processed/players_final.csv", index=False)
print(f"\n  Dataset base guardado: {len(tm)} jugadores")

# ─── PASO 2: Clustering ───
print("\n[2/5] Ejecutando clustering por posicion...")
from src.models.clustering import run_all_positions

df_clustered, cluster_metrics = run_all_positions(tm)

# ─── PASO 3: XGBoost + SHAP ───
print("\n[3/5] Entrenando XGBoost + calculando SHAP...")
from src.models.market_value import train_market_value_model, compute_shap_values, save_valued_dataset, REGRESSION_FEATURES

model, df_valued, feats, xgb_metrics = train_market_value_model(df_clustered)
X_shap = df_valued[feats].fillna(0)
shap_vals, chollos = compute_shap_values(model, X_shap, df_valued, feats)
save_valued_dataset(df_valued)

# ─── PASO 4: Embeddings + ChromaDB ───
print("\n[4/5] Generando embeddings e indexando en ChromaDB...")
from src.agents.embedding_agent import index_players_chromadb

# Merge clustered + valued
df_for_index = df_clustered.copy()
merge_cols = ["predicted_value_eur", "value_ratio"]
avail_merge = [c for c in merge_cols if c in df_valued.columns]
if avail_merge:
    df_for_index = df_for_index.merge(
        df_valued[["player", "squad"] + avail_merge],
        on=["player", "squad"], how="left", suffixes=("", "_val")
    )

collection, encoder = index_players_chromadb(df_for_index)

# ─── PASO 5: Test rápido ───
print("\n[5/5] Test rapido del sistema...")
from src.agents.embedding_agent import query_similar_players

test_queries = [
    "pivote defensivo con buena recuperacion de balones",
    "delantero goleador con alto xG",
    "defensa central que salga jugando",
]

for q in test_queries:
    results = query_similar_players(q, collection, encoder, n_results=3)
    print(f"\n  Query: '{q}'")
    for r in results:
        mv = r['market_value_eur']
        mv_str = f"EUR{mv/1e6:.1f}M" if mv > 0 else "N/A"
        print(f"    -> {r['player_name']} ({r['team']}) | Sim: {r['similarity_score']:.3f} | {mv_str}")

print("\n" + "=" * 60)
print("  Pipeline completado! Puedes correr: streamlit run app/main.py")
print("=" * 60)
