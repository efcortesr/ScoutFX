"""
llm_analyst.py — LLM Analyst Agent (Guía 04)
Genera justificaciones técnicas de scouting y análisis de chollos usando Gemini 2.5 Flash.
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
import os
import json
from dotenv import load_dotenv

load_dotenv()

LLM_ANALYST_SYSTEM_PROMPT = """Eres un analista técnico de fútbol con experiencia en data science deportivo.
Tu tarea es generar justificaciones técnicas detalladas para recomendaciones
de scouting y detección de chollos en el mercado de transferencias.

FORMATO DE RESPUESTA (siempre en este orden):
1. PERFIL DEL JUGADOR: nombre, club, liga, posición, edad, valor actual
2. POR QUÉ ENCAJA: explicación táctica basada en stats reales por 90 min
3. ANÁLISIS DE VALOR: predicción del modelo XGBoost, ratio valor/precio
4. FACTORES CLAVE (SHAP): los 3 features que más influyen en su valoración
5. VEREDICTO SCOUT: recomendación final en 2 líneas

REGLAS:
- Cita SIEMPRE números reales de las estadísticas proporcionadas
- Usa terminología técnica futbolística
- Si value_ratio > 2: destacar que es un CHOLLO con emoji 💎
- Si value_ratio < 0.5: advertir que puede estar SOBREVALORADO con ⚠️
- Máximo 250 palabras por jugador
- Responde SIEMPRE en español"""


def _get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        google_api_key=os.environ.get("GOOGLE_API_KEY", ""),
    )


def generate_scout_report(candidates: list, query: str, shap_data: dict = None) -> str:
    """Genera reporte técnico de scouting para candidatos."""
    llm = _get_llm()
    parts = []
    for i, c in enumerate(candidates[:5], 1):
        val = c.get("market_value_eur", 0)
        pred = c.get("predicted_value_eur", 0)
        ratio = c.get("value_ratio", 1)
        stats = c.get("stats", {})
        shap_info = ""
        if shap_data and c.get("player_name") in shap_data:
            shap_info = f"SHAP: {json.dumps(shap_data[c['player_name']], ensure_ascii=False)}"
        tag = "💎 CHOLLO" if ratio > 2 else ("⚠️ SOBREVALORADO" if ratio < 0.5 else "✅ Precio Justo")
        parts.append(f"""CANDIDATO {i}: {c.get('player_name','N/A')}
  Club: {c.get('team','')} | Liga: {c.get('league','')} | Pos: {c.get('position','')}
  Estilo: {c.get('cluster_label','N/A')}
  Valor: €{val/1e6:.1f}M → Predicho: €{pred/1e6:.1f}M | Ratio: {ratio:.2f}x {tag}
  Stats/90: PrgP={stats.get('prgp_per90','N/A')}, xAG={stats.get('xag_per90','N/A')}, xG={stats.get('xg_per90','N/A')}, Press%={stats.get('press_pct','N/A')}, Tkl+Int={stats.get('tkl_int_per90','N/A')}
  Similitud: {c.get('similarity_score',0):.3f} {shap_info}""")

    msgs = [
        SystemMessage(content=LLM_ANALYST_SYSTEM_PROMPT),
        HumanMessage(
            content=f'CONSULTA: "{query}"\n\nCANDIDATOS:\n'
            + "\n".join(parts)
            + "\n\nGenera el reporte técnico ordenado por relevancia."
        ),
    ]
    return llm.invoke(msgs).content


def generate_bargain_report(chollo: dict) -> str:
    """Genera justificación LLM para un chollo detectado."""
    llm = _get_llm()
    msgs = [
        SystemMessage(content=LLM_ANALYST_SYSTEM_PROMPT),
        HumanMessage(
            content=f"""Analiza este chollo:
JUGADOR: {chollo.get('player','N/A')} | Club: {chollo.get('squad','N/A')} | Liga: {chollo.get('league','N/A')}
Posición: {chollo.get('position_group','N/A')}
Valor actual: €{chollo.get('market_value_in_eur',0)/1e6:.1f}M
Valor predicho: €{chollo.get('predicted_value_eur',0)/1e6:.1f}M
Ratio: {chollo.get('value_ratio',1):.2f}x
Stats/90: xG={chollo.get('xG_per90','N/A')}, xAG={chollo.get('xAG_per90','N/A')}, PrgP={chollo.get('PrgP_per90','N/A')}, Press%={chollo.get('Press%','N/A')}
Genera una justificación técnica de por qué es un chollo."""
        ),
    ]
    return llm.invoke(msgs).content
