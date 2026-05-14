"""
preprocessor.py — Data Agent (Guía 01)
Limpieza, merge FBref + Transfermarkt, y feature engineering por posición.
Produce data/processed/players_final.csv que todos los agentes consumen.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

# Features específicas por posición (basadas en relevancia táctica)
FEATURES_BY_POSITION = {
    "GK": [
        "Save%", "CS%", "PSxG-GA", "Cmp%_Launch", "AvgLen_Launch",
        "Crosses_Stop%", "OPA_per90", "AvgDist",
    ],
    "CB": [
        "Tkl_per90", "Int_per90", "Clr_per90", "AerWon%",
        "Cmp%_Long", "PrgP_per90", "Blocks_per90", "Err",
    ],
    "FB": [  # Lateral
        "PrgC_per90", "CrsPA_per90", "TklWon%", "Att3rd_Tkl%",
        "xAG_per90", "KP_per90", "PrgDist_per90",
    ],
    "MF": [  # Pivote/mediocentro
        "PrgP_per90", "PrgR_per90", "Cmp%_Med", "Press%",
        "Tkl+Int_per90", "Carries_PrgDist_per90", "SCA_per90", "xAG_per90",
    ],
    "AM": [  # Mediapunta/extremo
        "xAG_per90", "SCA_per90", "KP_per90", "xG_per90",
        "Carries_1/3_per90", "Succ%_Drib", "G+A_per90",
    ],
    "FW": [  # Delantero
        "xG_per90", "Sh_per90", "SoT%", "G_per90",
        "Succ%_Drib", "Carries_1/3_per90", "AerWon%", "xAG_per90",
    ],
}


def engineer_features(fbref_df, tm_df, output_path="data/processed/players_final.csv"):
    """
    Merge FBref + Transfermarkt. Feature engineering por posición.
    Normaliza con StandardScaler por grupo de posición.

    Args:
        fbref_df: DataFrame de FBref con stats de jugadores
        tm_df: DataFrame de Transfermarkt con valores de mercado
        output_path: ruta del CSV final

    Returns:
        DataFrame procesado y normalizado
    """
    # --- JOIN FBref + Transfermarkt por nombre limpio ---
    if hasattr(fbref_df.index, "get_level_values"):
        try:
            fbref_df = fbref_df.reset_index()
        except Exception:
            pass

    # Limpiar nombre del jugador para matching
    player_col = "player" if "player" in fbref_df.columns else fbref_df.columns[0]
    fbref_df["name_clean"] = (
        fbref_df[player_col]
        .astype(str)
        .str.lower()
        .str.strip()
        .str.replace(r"[^a-z\s]", "", regex=True)
    )

    merged = pd.merge(
        fbref_df,
        tm_df[["name_clean", "market_value_in_eur", "player_id"]],
        on="name_clean",
        how="left",
    )

    print(f"📊 Merge FBref-TM: {len(merged)} registros totales")
    match_count = merged["market_value_in_eur"].notna().sum()
    match_rate = match_count / len(merged) * 100
    print(f"   Match rate: {match_count}/{len(merged)} ({match_rate:.1f}%)")

    # --- Filtrar mínimo minutos jugados (≥900 min = ~10 partidos) ---
    if "Min" in merged.columns:
        before = len(merged)
        merged = merged[merged["Min"] >= 900].copy()
        print(f"   Filtro ≥900 min: {before} → {len(merged)} jugadores")

    # --- Crear features per90 si no existen ---
    per90_cols = ["Tkl", "Int", "Clr", "PrgP", "PrgR", "PrgC", "xG", "xAG", "SCA", "KP"]
    for col in per90_cols:
        if col in merged.columns and f"{col}_per90" not in merged.columns:
            merged[f"{col}_per90"] = merged[col] / (merged["Min"] / 90)

    # --- Detectar posición ---
    if "Pos" in merged.columns and "position_group" not in merged.columns:
        merged["position_group"] = merged["Pos"].apply(_map_position)

    # --- Normalizar por posición ---
    scaled_dfs = []
    for pos, features in FEATURES_BY_POSITION.items():
        mask = merged["position_group"] == pos
        if mask.sum() == 0:
            continue
        subset = merged[mask].copy()

        available = [f for f in features if f in subset.columns]
        if len(available) < 3:
            # Guardar de todas formas sin normalizar
            subset["features_used"] = str(available)
            subset["position_group"] = pos
            scaled_dfs.append(subset)
            continue

        scaler = StandardScaler()
        subset[available] = scaler.fit_transform(subset[available].fillna(0))
        subset["position_group"] = pos
        subset["features_used"] = str(available)
        scaled_dfs.append(subset)

    if not scaled_dfs:
        print("⚠️ No se pudieron generar features. Guardando merge sin normalizar.")
        merged.to_csv(output_path, index=False)
        return merged

    final = pd.concat(scaled_dfs, ignore_index=True)

    # Guardar
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final.to_csv(output_path, index=False)
    print(f"✅ Dataset final: {len(final)} jugadores listos para ML → {output_path}")

    return final


def _map_position(pos_str):
    """
    Mapea posiciones de FBref a grupos simplificados.
    FBref usa: GK, DF, MF, FW (y combinaciones como DF,MF)
    """
    if pd.isna(pos_str):
        return "MF"

    pos = str(pos_str).upper()

    if "GK" in pos:
        return "GK"
    elif "FW" in pos and "MF" in pos:
        return "AM"  # Mediapunta/extremo
    elif "FW" in pos:
        return "FW"
    elif "MF" in pos and "DF" in pos:
        return "MF"  # Mediocentro defensivo
    elif "MF" in pos:
        return "MF"
    elif "DF" in pos:
        # Intentar distinguir CB vs FB basándose en stats si están disponibles
        return "CB"  # Default a CB, se podría refinar
    else:
        return "MF"


if __name__ == "__main__":
    from scraper_fbref import scrape_fbref_stats
    from loader_transfermarkt import load_transfermarkt

    fbref = scrape_fbref_stats()
    tm = load_transfermarkt()
    final = engineer_features(fbref, tm)
    print(f"\nDistribución por posición:")
    print(final["position_group"].value_counts())
