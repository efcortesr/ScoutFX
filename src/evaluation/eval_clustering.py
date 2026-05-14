"""
eval_clustering.py — Evaluación de Clustering (Guía 02)
Métricas: Silhouette Score + Davies-Bouldin por posición.
"""
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score

def evaluate_clustering(df):
    """Evalúa la calidad del clustering por posición."""
    from src.models.clustering import CLUSTER_FEATURES
    results = []
    for pos, features in CLUSTER_FEATURES.items():
        subset = df[df["position_group"]==pos]
        if len(subset)<10 or "cluster_kmeans" not in subset.columns:
            continue
        avail = [f for f in features if f in subset.columns]
        if len(avail)<2: continue
        X = subset[avail].fillna(0).values
        labels = subset["cluster_kmeans"].values
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters < 2: continue
        sil = silhouette_score(X, labels)
        db = davies_bouldin_score(X, labels)
        results.append({"position":pos, "n_players":len(subset), "n_clusters":n_clusters,
                        "silhouette":round(sil,3), "davies_bouldin":round(db,3),
                        "pass_sil": "✅" if sil>0.25 else "❌",
                        "pass_db": "✅" if db<1.5 else "❌"})
    df_res = pd.DataFrame(results)
    print("\n📊 Evaluación de Clustering:")
    print(df_res.to_string(index=False))
    return df_res

if __name__ == "__main__":
    df = pd.read_csv("data/processed/players_clustered.csv")
    evaluate_clustering(df)
