"""
scout_rag_agent.py — RAG Scout Agent (Guía 04)
Agente ReAct con LangChain + Gemini 2.5 Flash que busca jugadores, chollos y SHAP values.
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import Tool
from langgraph.prebuilt import create_react_agent
import pandas as pd
import json
import os
from dotenv import load_dotenv

load_dotenv()


def search_similar_players_tool(query_json: str) -> str:
    """Tool para buscar jugadores similares en ChromaDB."""
    from src.agents.embedding_agent import query_similar_players, load_resources

    collection, encoder, _ = load_resources()
    try:
        params = json.loads(query_json)
    except Exception:
        params = {"query": query_json, "position": None, "max_price_eur": None, "n": 5}
    candidates = query_similar_players(
        query_text=params.get("query", ""),
        collection=collection,
        encoder=encoder,
        n_results=params.get("n", 5),
        position_filter=params.get("position"),
        max_price_eur=params.get("max_price_eur"),
    )
    return json.dumps(candidates, ensure_ascii=False, indent=2)


def get_bargains_tool(position: str) -> str:
    """Tool para obtener chollos detectados por XGBoost."""
    chollos = pd.read_csv("data/processed/chollos_detectados.csv")
    if position.upper() != "ALL":
        chollos = chollos[chollos["position_group"] == position.upper()]
    cols = [
        "player", "squad", "league", "position_group", "market_value_in_eur",
        "predicted_value_eur", "value_ratio", "xG_per90", "xAG_per90", "PrgP_per90",
    ]
    avail = [c for c in cols if c in chollos.columns]
    return chollos.nlargest(10, "value_ratio")[avail].to_json(orient="records", force_ascii=False)


def get_player_shap_tool(player_name: str) -> str:
    """Tool para obtener SHAP values de un jugador."""
    from src.agents.embedding_agent import load_resources

    _, _, shap_df = load_resources()
    players_df = pd.read_csv("data/processed/players_valued.csv")
    match = players_df[
        players_df["player"].str.lower().str.contains(player_name.lower(), na=False)
    ]
    if match.empty:
        return json.dumps({"error": f"Jugador '{player_name}' no encontrado"})
    idx = match.index[0]
    shap_row = shap_df.loc[idx] if idx in shap_df.index else pd.Series()
    shap_detail = {}
    if len(shap_row) > 0:
        top_shap = shap_row.abs().nlargest(5)
        shap_detail = {
            f: {
                "shap_value": float(shap_row[f]),
                "direction": "positivo" if shap_row[f] > 0 else "negativo",
            }
            for f in top_shap.index
            if f in shap_row.index
        }
    pr = match.iloc[0]
    return json.dumps(
        {
            "player_name": pr["player"],
            "team": pr.get("squad", ""),
            "market_value_eur": float(pr.get("market_value_in_eur", 0)),
            "predicted_value_eur": float(pr.get("predicted_value_eur", 0)),
            "value_ratio": float(pr.get("value_ratio", 1)),
            "top_shap_features": shap_detail,
        },
        ensure_ascii=False,
        indent=2,
    )


def create_scout_agent():
    """Crea el agente Scout usando ReAct con LangGraph + Gemini 2.5 Flash."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1,
        google_api_key=os.environ.get("GOOGLE_API_KEY", ""),
    )
    tools = [
        Tool(
            name="search_similar_players",
            func=search_similar_players_tool,
            description=(
                'Busca jugadores similares en ChromaDB. Input JSON: '
                '{"query":str, "position":str|null, "max_price_eur":float|null, "n":int}'
            ),
        ),
        Tool(
            name="get_bargains",
            func=get_bargains_tool,
            description="Obtiene chollos detectados por XGBoost. Input: posición (MF/FW/CB/FB/GK/ALL)",
        ),
        Tool(
            name="get_player_shap",
            func=get_player_shap_tool,
            description="Obtiene SHAP values de un jugador. Input: nombre del jugador.",
        ),
    ]
    return create_react_agent(llm, tools=tools)


if __name__ == "__main__":
    executor = create_scout_agent()
    result = executor.invoke({"messages": [("user", "Busca pivotes defensivos baratos similares a Busquets")]})
    print(result["messages"][-1].content)
