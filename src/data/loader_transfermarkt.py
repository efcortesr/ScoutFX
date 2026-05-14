"""
loader_transfermarkt.py — Data Agent (Guía 01)
Carga el dataset de Transfermarkt desde Kaggle.
Dataset: https://www.kaggle.com/datasets/davidcariboo/player-scores
"""

import pandas as pd


def load_transfermarkt(path="data/raw/players.csv"):
    """
    Carga el dataset de Transfermarkt con valor de mercado histórico.

    Args:
        path: ruta al archivo players.csv de Kaggle

    Returns:
        DataFrame limpio con el valor de mercado más reciente por jugador
    """
    print(f"📥 Cargando Transfermarkt desde {path}...")

    tm = pd.read_csv(path)

    # Detectar columna de nombre del jugador
    if "name" in tm.columns and "player_name" not in tm.columns:
        tm["player_name"] = tm["name"]
    elif "first_name" in tm.columns and "last_name" in tm.columns and "player_name" not in tm.columns:
        tm["player_name"] = tm["first_name"].fillna("") + " " + tm["last_name"].fillna("")
        tm["player_name"] = tm["player_name"].str.strip()

    # Filtrar solo jugadores con valor de mercado > 0
    tm = tm[tm["market_value_in_eur"] > 0].copy()

    # Tomar el valor de mercado más reciente por jugador
    if "date" in tm.columns:
        tm = tm.sort_values("date").groupby("player_id").last().reset_index()

    # Limpiar nombres para el join con FBref
    tm["name_clean"] = (
        tm["player_name"]
        .str.lower()
        .str.strip()
        .str.replace(r"[^a-z\s]", "", regex=True)
    )

    print(f"✅ Transfermarkt: {len(tm)} jugadores con valor de mercado")
    return tm


if __name__ == "__main__":
    df = load_transfermarkt()
    print(f"\nColumnas: {df.columns.tolist()}")
    print(f"\nDistribución de valores (en millones €):")
    print((df["market_value_in_eur"] / 1e6).describe())
