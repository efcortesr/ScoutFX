"""
scraper_fbref.py — Data Agent (Guía 01)
Descarga estadísticas de jugadores desde FBref usando soccerdata.
Cubre las 5 grandes ligas europeas, temporada 23/24.
"""

import soccerdata as sd
import pandas as pd
import time
import os


def scrape_fbref_stats(
    leagues=None,
    seasons=None,
    output_path="data/raw/fbref_stats.csv"
):
    """
    Descarga estadísticas por 90 minutos de FBref para +2500 jugadores.
    Incluye: pases, presión, posesión, duelos, portería, etc.

    Args:
        leagues: lista de ligas a descargar
        seasons: lista de temporadas
        output_path: ruta donde guardar el CSV crudo

    Returns:
        DataFrame con stats de jugadores
    """
    if leagues is None:
        leagues = [
            "ENG-Premier League",
            "ESP-La Liga",
            "GER-Bundesliga",
            "ITA-Serie A",
            "FRA-Ligue 1",
        ]
    if seasons is None:
        seasons = ["2324"]

    print(f"📥 Descargando datos FBref para {len(leagues)} ligas, temporada(s) {seasons}...")
    ws = sd.FBref(leagues=leagues, seasons=seasons)

    stat_types = ["passing", "misc", "defense", "shooting"]
    dfs = []

    for stat_type in stat_types:
        print(f"  → Descargando stats: {stat_type}...")
        try:
            df = ws.read_player_season_stats(stat_type=stat_type)
            dfs.append(df)
            print(f"    ✅ {stat_type}: {len(df)} registros")
        except Exception as e:
            print(f"    ⚠️ Error en {stat_type}: {e}")
        # Rate limit: FBref bloquea scraping rápido
        time.sleep(3)

    if not dfs:
        raise RuntimeError("No se pudieron descargar datos de FBref.")

    # Merge por jugador-temporada (join por índice)
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.join(df, how="outer", rsuffix="_dup")

    # Eliminar columnas duplicadas
    merged = merged.loc[:, ~merged.columns.str.endswith("_dup")]

    # Asegurar directorio de salida
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged.to_csv(output_path)

    print(f"✅ FBref: {len(merged)} jugadores descargados → {output_path}")
    return merged


if __name__ == "__main__":
    df = scrape_fbref_stats()
    print(f"\nColumnas disponibles ({len(df.columns)}):")
    print(df.columns.tolist())
    print(f"\nPrimeras filas:")
    print(df.head())
