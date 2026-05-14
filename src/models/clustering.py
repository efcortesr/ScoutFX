"""
clustering.py — ML Agent (Guía 02)
Clustering táctico: KMeans + HDBSCAN + UMAP 2D por posición.
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import hdbscan
import umap
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
import os

CLUSTER_FEATURES = {
    "MF": ["PrgP_per90","Press%","Tkl+Int_per90","Cmp%_Med","xAG_per90","PrgR_per90","Carries_PrgDist_per90"],
    "FW": ["xG_per90","Sh_per90","SoT%","G_per90","Succ%_Drib","xAG_per90","Carries_1/3_per90"],
    "CB": ["Tkl_per90","Int_per90","Clr_per90","AerWon%","Cmp%_Long","PrgP_per90","Blocks_per90"],
    "FB": ["PrgC_per90","CrsPA_per90","TklWon%","xAG_per90","KP_per90","Att3rd_Tkl%"],
}

LABEL_TEMPLATES = {
    "MF": {"PrgP_per90":"Distribuidor","Press%":"Presionador","Tkl+Int_per90":"Pivot Defensivo","xAG_per90":"Creador","PrgR_per90":"Box-to-Box"},
    "FW": {"xG_per90":"Finalizador","Succ%_Drib":"Regateador","xAG_per90":"Delantero Creador","Sh_per90":"Artillero"},
    "CB": {"Tkl_per90":"Defensa Activo","PrgP_per90":"Defensa Progresivo","AerWon%":"Defensa Aéreo"},
    "FB": {"PrgC_per90":"Lateral Profundo","xAG_per90":"Lateral Asistidor","TklWon%":"Lateral Defensivo"},
}

def find_optimal_k(X, k_range=range(2,9)):
    silhouettes = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        silhouettes.append(silhouette_score(X, labels))
    best_k = list(k_range)[np.argmax(silhouettes)]
    print(f"  Mejor k={best_k} (Sil={max(silhouettes):.3f})")
    return best_k

def cluster_by_position(df, position="MF"):
    features = CLUSTER_FEATURES.get(position, CLUSTER_FEATURES["MF"])
    subset = df[df["position_group"]==position].copy()
    if len(subset)<10:
        return subset, 0.0, float("inf")
    avail = [f for f in features if f in subset.columns]
    if len(avail)<2:
        return subset, 0.0, float("inf")
    X = subset[avail].fillna(0).values
    print(f"\n🔵 Clustering {position}: {len(subset)} jugadores, {len(avail)} features")
    max_k = min(8, len(subset)-1)
    best_k = find_optimal_k(X, range(2, max(3,max_k+1)))
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    subset["cluster_kmeans"] = km.fit_predict(X)
    sil = silhouette_score(X, subset["cluster_kmeans"])
    db = davies_bouldin_score(X, subset["cluster_kmeans"])
    print(f"  KMeans — Sil: {sil:.3f} | DB: {db:.3f}")
    try:
        hdb = hdbscan.HDBSCAN(min_cluster_size=max(10,len(subset)//20), min_samples=5)
        subset["cluster_hdbscan"] = hdb.fit_predict(X)
    except:
        subset["cluster_hdbscan"] = -1
    try:
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15,len(subset)-1), min_dist=0.1)
        emb = reducer.fit_transform(X)
        subset["umap_x"] = emb[:,0]
        subset["umap_y"] = emb[:,1]
    except:
        subset["umap_x"] = 0.0; subset["umap_y"] = 0.0; reducer = None
    templates = LABEL_TEMPLATES.get(position,{})
    centroids = subset.groupby("cluster_kmeans")[avail].mean()
    labels = {}
    used = set()
    for cid, row in centroids.iterrows():
        lbl = templates.get(row.idxmax(), f"Estilo {cid}")
        if lbl in used:
            lbl = f"{lbl} {cid}"
        used.add(lbl); labels[cid] = lbl
    subset["cluster_label"] = subset["cluster_kmeans"].map(labels)
    os.makedirs("models/checkpoints", exist_ok=True)
    joblib.dump(km, f"models/checkpoints/kmeans_{position}.pkl")
    if reducer: joblib.dump(reducer, f"models/checkpoints/umap_{position}.pkl")
    return subset, sil, db

def run_all_positions(df):
    all_res = []; metrics = {}
    for pos in CLUSTER_FEATURES:
        if pos not in df["position_group"].unique(): continue
        c, s, d = cluster_by_position(df, pos)
        all_res.append(c); metrics[pos] = {"silhouette":s,"davies_bouldin":d,"n":len(c)}
    if all_res:
        result = pd.concat(all_res, ignore_index=True)
        os.makedirs("data/processed", exist_ok=True)
        result.to_csv("data/processed/players_clustered.csv", index=False)
        print(f"\n✅ Clustering completo: {len(result)} jugadores")
        return result, metrics
    return pd.DataFrame(), metrics

if __name__ == "__main__":
    df = pd.read_csv("data/processed/players_final.csv")
    run_all_positions(df)
