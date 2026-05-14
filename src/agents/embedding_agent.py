"""
embedding_agent.py — Embedding Agent (Guía 03)
Convierte perfiles de jugadores en texto semántico, genera embeddings
con sentence-transformers, y los indexa en ChromaDB.
"""
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm
import json
import os

STYLE_DESCRIPTIONS = {
    "Pivot Defensivo": "Es un centrocampista con perfil defensivo, excelente en recuperar balones y cortar líneas de pase. Similar a Busquets o Casemiro.",
    "Distribuidor": "Es un centrocampista distribuidor de juego, destacado en pases progresivos y control del tempo. Similar a Toni Kroos o Frenkie de Jong.",
    "Box-to-Box": "Es un centrocampista completo que contribuye en ambas fases. Similar a Kanté o Declan Rice.",
    "Creador": "Es un centrocampista creativo con alta participación en goles. Similar a De Bruyne o Ødegaard.",
    "Finalizador": "Es un delantero centro letal de cara a gol, con alto xG. Similar a Haaland o Harry Kane.",
    "Defensa Activo": "Es un defensa central agresivo en anticipaciones. Similar a Van Dijk o Saliba.",
    "Defensa Progresivo": "Es un defensa que contribuye en la salida de balón. Similar a Rúben Dias o Bastoni.",
    "Presionador": "Es un centrocampista intenso en la presión, que recupera balones en campo rival.",
    "Regateador": "Es un delantero habilidoso con el balón, capaz de desbordar en el uno contra uno.",
    "Delantero Creador": "Es un delantero con capacidad de asistir además de marcar goles.",
}

def build_player_document(row):
    """Convierte stats de un jugador en texto semántico rico para embeddings."""
    name = row.get("player","Unknown"); team = row.get("squad","Unknown")
    league = row.get("league","Unknown"); pos = row.get("position_group","MF")
    age = int(row.get("Age",0)) if not pd.isna(row.get("Age",0)) else "N/A"
    cluster_label = row.get("cluster_label","jugador de campo")
    mv = row.get("market_value_in_eur",0)
    prgp = round(row.get("PrgP_per90",0),2); xag = round(row.get("xAG_per90",0),2)
    xg = round(row.get("xG_per90",0),2); press = round(row.get("Press%",0),1)
    tkl = round(row.get("Tkl+Int_per90",0),2); mins = int(row.get("Min",0) or 0)
    vs = f"€{mv/1e6:.1f}M" if mv and mv>=1e6 else (f"€{mv/1e3:.0f}K" if mv and mv>0 else "N/A")
    doc = (f"{name} es un {cluster_label} que juega como {pos} en {team} ({league}). "
           f"Tiene {age} años y valor de mercado de {vs}. {mins} minutos esta temporada. "
           f"Stats por 90min: pases progresivos ({prgp}), xAG ({xag}), xG ({xg}), "
           f"presión ({press}%), tackles+int ({tkl}). ")
    doc += STYLE_DESCRIPTIONS.get(cluster_label,"")
    return doc

def build_player_metadata(row):
    """Metadata dict para ChromaDB."""
    def safe_float(val, default=0.0):
        try: return float(val) if val is not None and not (isinstance(val,float) and np.isnan(val)) else default
        except: return default
    return {
        "player_name": str(row.get("player","")), "team": str(row.get("squad","")),
        "league": str(row.get("league","")), "position": str(row.get("position_group","")),
        "cluster_label": str(row.get("cluster_label","")),
        "cluster_id": int(row.get("cluster_kmeans",-1) or -1),
        "age": safe_float(row.get("Age")), "market_value_eur": safe_float(row.get("market_value_in_eur")),
        "predicted_value_eur": safe_float(row.get("predicted_value_eur")),
        "value_ratio": safe_float(row.get("value_ratio"),1.0),
        "prgp_per90": safe_float(row.get("PrgP_per90")), "xag_per90": safe_float(row.get("xAG_per90")),
        "xg_per90": safe_float(row.get("xG_per90")), "press_pct": safe_float(row.get("Press%")),
        "tkl_int_per90": safe_float(row.get("Tkl+Int_per90")),
        "minutes": safe_float(row.get("Min")),
        "umap_x": safe_float(row.get("umap_x")), "umap_y": safe_float(row.get("umap_y")),
    }

def index_players_chromadb(df, model_name="all-MiniLM-L6-v2", persist_dir="models/checkpoints/chromadb"):
    """Genera embeddings e indexa en ChromaDB."""
    print(f"\n🧬 Cargando modelo de embeddings: {model_name}")
    encoder = SentenceTransformer(model_name)
    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)
    try: client.delete_collection("football_players")
    except: pass
    collection = client.create_collection(name="football_players", metadata={"hnsw:space":"cosine"})
    batch_size = 100; total = len(df)
    print(f"📥 Indexando {total} jugadores en ChromaDB...")
    all_docs = []
    for i in tqdm(range(0, total, batch_size)):
        batch = df.iloc[i:i+batch_size]
        documents = [build_player_document(row) for _, row in batch.iterrows()]
        metadatas = [build_player_metadata(row) for _, row in batch.iterrows()]
        ids = [f"player_{i+j}" for j in range(len(batch))]
        embeddings = encoder.encode(documents, show_progress_bar=False, normalize_embeddings=True).tolist()
        collection.add(documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids)
        all_docs.extend(documents)
    # Guardar documentos para auditoría
    os.makedirs("data/processed", exist_ok=True)
    pd.DataFrame({"document": all_docs}).to_csv("data/processed/player_documents.csv", index=False)
    print(f"\n✅ ChromaDB: {collection.count()} jugadores indexados en {persist_dir}")
    return collection, encoder

def load_resources(persist_dir="models/checkpoints/chromadb"):
    """Carga ChromaDB + encoder + SHAP para uso por otros agentes."""
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_collection("football_players")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    shap_path = "data/processed/shap_values.csv"
    shap_df = pd.read_csv(shap_path, index_col=0) if os.path.exists(shap_path) else pd.DataFrame()
    return collection, encoder, shap_df

def query_similar_players(query_text, collection, encoder, n_results=5,
                           position_filter=None, max_price_eur=None, leagues=None):
    """Busca los N jugadores más similares a una consulta en lenguaje natural."""
    conditions = []
    if position_filter: 
        conditions.append({"position": {"$eq": position_filter}})
    if max_price_eur: 
        conditions.append({"market_value_eur": {"$lte": float(max_price_eur)}})
    if leagues and len(leagues) > 0:
        if len(leagues) == 1:
            conditions.append({"league": {"$eq": leagues[0]}})
        else:
            conditions.append({"league": {"$in": leagues}})
            
    if len(conditions) == 1:
        where_filter = conditions[0]
    elif len(conditions) > 1:
        where_filter = {"$and": conditions}
    else:
        where_filter = None

    query_emb = encoder.encode([query_text], normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=query_emb, n_results=n_results,
                                where=where_filter,
                                include=["documents","metadatas","distances"])
    candidates = []
    for j in range(len(results["ids"][0])):
        meta = results["metadatas"][0][j]; dist = results["distances"][0][j]
        candidates.append({
            "player_name": meta["player_name"], "team": meta["team"],
            "league": meta["league"], "position": meta["position"],
            "cluster_label": meta["cluster_label"],
            "market_value_eur": meta["market_value_eur"],
            "predicted_value_eur": meta["predicted_value_eur"],
            "value_ratio": meta["value_ratio"],
            "similarity_score": round(1-dist, 4),
            "stats": {"prgp_per90":meta["prgp_per90"],"xag_per90":meta["xag_per90"],
                       "xg_per90":meta["xg_per90"],"press_pct":meta["press_pct"],
                       "tkl_int_per90":meta["tkl_int_per90"]},
            "document": results["documents"][0][j],
        })
    return candidates

if __name__ == "__main__":
    df = pd.read_csv("data/processed/players_clustered.csv")
    valued = pd.read_csv("data/processed/players_valued.csv")
    df = df.merge(valued[["player","squad","predicted_value_eur","value_ratio"]],
                   on=["player","squad"], how="left")
    collection, encoder = index_players_chromadb(df)
    print("\n🔍 Test:")
    for r in query_similar_players("pivote defensivo con alta recuperación", collection, encoder, 3, "MF"):
        print(f"  → {r['player_name']} ({r['team']}) — Sim: {r['similarity_score']:.3f}")
