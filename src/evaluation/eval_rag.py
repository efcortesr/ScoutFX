"""
eval_rag.py — Evaluación del Sistema RAG (Guía 04)
20 queries de test con respuesta esperada. OBLIGATORIO para el informe.
"""
import json

TEST_QUERIES = [
    {"query":"pivote defensivo bueno en recuperación de balones, liga española",
     "expected_position":"MF","expected_cluster":"Pivot Defensivo","expected_league":"ESP-La Liga",
     "check": lambda r: r[0]["stats"]["tkl_int_per90"]>2.5 if r else False},
    {"query":"delantero centro con alto xG y buen juego aéreo",
     "expected_position":"FW","expected_cluster":"Finalizador",
     "check": lambda r: r[0]["stats"]["xg_per90"]>0.3 if r else False},
    {"query":"lateral ofensivo con muchos centros y asistencias",
     "expected_position":"FB",
     "check": lambda r: r[0]["stats"]["xag_per90"]>0.05 if r else False},
    {"query":"mediocentro distribuidor de juego tipo Toni Kroos",
     "expected_position":"MF","expected_cluster":"Distribuidor",
     "check": lambda r: r[0]["stats"]["prgp_per90"]>3 if r else False},
    {"query":"extremo creativo con regates, precio menor a 10 millones",
     "expected_position":"AM","max_price":10_000_000,
     "check": lambda r: r[0]["market_value_eur"]<=10_000_000 if r else False},
    {"query":"defensa central progresivo con buen pase largo",
     "expected_position":"CB","expected_cluster":"Defensa Progresivo",
     "check": lambda r: r[0]["position"]=="CB" if r else False},
    {"query":"delantero regateador que pueda jugar por banda",
     "expected_position":"FW",
     "check": lambda r: len(r)>0},
    {"query":"centrocampista box-to-box con buena presión y tackles",
     "expected_position":"MF",
     "check": lambda r: r[0]["stats"]["tkl_int_per90"]>1.5 if r else False},
    {"query":"delantero barato con alto ratio de gol, menos de 5 millones",
     "expected_position":"FW","max_price":5_000_000,
     "check": lambda r: r[0]["market_value_eur"]<=5_000_000 if r else False},
    {"query":"mediocentro creativo de la Premier League con muchas asistencias esperadas",
     "expected_position":"MF","expected_league":"ENG-Premier League",
     "check": lambda r: r[0]["stats"]["xag_per90"]>0.1 if r else False},
    {"query":"defensa central aéreo dominante de la Serie A",
     "expected_position":"CB","expected_league":"ITA-Serie A",
     "check": lambda r: r[0]["position"]=="CB" if r else False},
    {"query":"lateral derecho defensivo de la Bundesliga",
     "expected_position":"FB","expected_league":"GER-Bundesliga",
     "check": lambda r: r[0]["position"]=="FB" if r else False},
    {"query":"jugador joven mediocampista con proyección, sub-23",
     "expected_position":"MF",
     "check": lambda r: len(r)>0},
    {"query":"delantero centro goleador de la Ligue 1",
     "expected_position":"FW","expected_league":"FRA-Ligue 1",
     "check": lambda r: r[0]["stats"]["xg_per90"]>0.2 if r else False},
    {"query":"pivote que sea chollo de mercado, valor predicho mayor al actual",
     "expected_position":"MF",
     "check": lambda r: r[0]["value_ratio"]>1.5 if r else False},
    {"query":"centrocampista presionador intenso estilo Gegenpressing",
     "expected_position":"MF",
     "check": lambda r: r[0]["stats"]["press_pct"]>25 if r else False},
    {"query":"defensa central que salga jugando con el balón desde atrás",
     "expected_position":"CB",
     "check": lambda r: r[0]["position"]=="CB" if r else False},
    {"query":"delantero que cree juego y asista además de marcar",
     "expected_position":"FW","expected_cluster":"Delantero Creador",
     "check": lambda r: r[0]["stats"]["xag_per90"]>0.1 if r else False},
    {"query":"lateral izquierdo ofensivo similar a Theo Hernández",
     "expected_position":"FB",
     "check": lambda r: len(r)>0},
    {"query":"chollos en cualquier posición de ligas europeas",
     "expected_position":None,
     "check": lambda r: len(r)>0},
]

def evaluate_rag_system(query_func, n_queries=20):
    """Evalúa el sistema RAG con las queries de test."""
    results = {"total":min(n_queries,len(TEST_QUERIES)),"passed":0,"failed":0,"details":[]}
    for i, test in enumerate(TEST_QUERIES[:n_queries]):
        try:
            pos = test.get("expected_position")
            max_p = test.get("max_price")
            candidates = query_func(test["query"], position_filter=pos, max_price_eur=max_p)
            passed = test["check"](candidates) if candidates else False
            if passed: results["passed"] += 1
            else: results["failed"] += 1
            results["details"].append({"query":test["query"],"passed":passed,
                "n_results":len(candidates) if candidates else 0,
                "top_player":candidates[0]["player_name"] if candidates else "N/A"})
        except Exception as e:
            results["failed"] += 1
            results["details"].append({"query":test["query"],"passed":False,"error":str(e)})
    acc = results["passed"]/results["total"]*100 if results["total"]>0 else 0
    print(f"\n📊 Evaluación RAG: {acc:.1f}% ({results['passed']}/{results['total']})")
    for d in results["details"]:
        icon = "✅" if d["passed"] else "❌"
        print(f"  {icon} {d['query'][:60]}... → {d.get('top_player','N/A')}")
    return results

if __name__ == "__main__":
    from src.agents.embedding_agent import query_similar_players, load_resources
    collection, encoder, _ = load_resources()
    def qf(q, position_filter=None, max_price_eur=None):
        return query_similar_players(q, collection, encoder, 5, position_filter, max_price_eur)
    evaluate_rag_system(qf)
