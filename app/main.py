"""
main.py — Frontend Agent (Guía 05)
App Streamlit con 4 tabs: Scout IA, Detector de Chollos, Mapa Táctico, EDA.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import sys

# Agregar src al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Inyectar variables de entorno desde st.secrets para Streamlit Cloud
try:
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception:
    pass

from src.agents.embedding_agent import query_similar_players
from src.agents.llm_analyst import generate_scout_report, generate_bargain_report
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# ─── CONFIGURACIÓN ───
st.set_page_config(page_title="ScoutFX — Scout + Valuador", page_icon="⚽",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .metric-card {background:#f0f4f8;border-radius:12px;padding:1rem;border-left:4px solid #003d79;}
    .chollo-badge {background:#27ae60;color:white;padding:3px 10px;border-radius:20px;font-size:0.8rem;font-weight:bold;}
    .stTabs [data-baseweb="tab"] {font-size:1rem;font-weight:600;}
</style>
""", unsafe_allow_html=True)

# ─── CARGA DE DATOS ───
FC_POS_MAP = {"GK": "POR", "CB": "DFC", "FB": "LI/LD", "MF": "MC", "AM": "MCO", "FW": "DC"}
FC_POS_MAP_INV = {v: k for k, v in FC_POS_MAP.items()}

@st.cache_data
def load_data():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        df_c = pd.read_csv(os.path.join(base,"data/processed/players_clustered.csv"))
        df_v = pd.read_csv(os.path.join(base,"data/processed/players_valued.csv"))
        df_ch = pd.read_csv(os.path.join(base,"data/processed/chollos_detectados.csv"))
        df = df_c.merge(df_v[["player","squad","predicted_value_eur","value_ratio"]],
                        on=["player","squad"], how="left", suffixes=("","_val"))
        
        # Mapear a posiciones de FC
        for d in [df, df_v, df_ch]:
            if "position_group" in d.columns:
                d["position_group"] = d["position_group"].replace(FC_POS_MAP)
                
        return df, df_v, df_ch
    except FileNotFoundError as e:
        st.error(f"⚠️ Datos no encontrados: {e}")
        st.info("Ejecuta primero los scripts del pipeline para generar los datos.")
        st.stop()

@st.cache_resource
def load_agents():
    from src.agents.embedding_agent import load_resources
    from src.agents.scout_rag_agent import create_scout_agent
    collection, encoder, shap_df = load_resources()
    scout_executor = create_scout_agent()
    # Optimización: inicializar el extractor de parámetros una sola vez
    extractor = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    return collection, encoder, shap_df, scout_executor, extractor

# ─── SIDEBAR ───
with st.sidebar:
    st.title("⚽ ScoutFX")
    st.caption("Sistema de Scouting e Inteligencia de Mercado")
    st.divider()
    st.markdown("**Filtros Globales**")
    position_filter = st.selectbox("Posición", ["Todas", "POR", "DFC", "LI/LD", "MC", "MCO", "DC"])
    LEAGUE_OPTIONS = {
        "ENG-Premier League": "GB1", "ESP-La Liga": "ES1",
        "GER-Bundesliga": "L1", "ITA-Serie A": "IT1",
        "FRA-Ligue 1": "FR1", "BRA-Brasileirão": "BRA1",
        "ARG-Primera": "ARG1", "POR-Primeira Liga": "PO1", "NED-Eredivisie": "NL1"
    }
    league_filter_names = st.multiselect("Ligas", list(LEAGUE_OPTIONS.keys()),
        default=["ENG-Premier League","ESP-La Liga","GER-Bundesliga","ITA-Serie A","FRA-Ligue 1"])
    league_filter = [LEAGUE_OPTIONS[k] for k in league_filter_names]
    max_price = st.slider("Precio máximo (M€)", 0, 150, 50, 5) * 1_000_000
    st.divider()
    st.caption("Datos: FBref + Transfermarkt | Temporada 23/24")
    st.caption("Modelo: XGBoost + SHAP | LLM: Gemini 2.5 Flash")

# ─── HEADER ───
st.title("⚽ ScoutFX")
st.subheader("Sistema de Scouting Inteligente + Detector de Chollos")
df, df_valued, df_chollos = load_data()

c1,c2,c3,c4 = st.columns(4)
with c1: st.metric("🧑‍🤝‍🧑 Jugadores", f"{len(df):,}")
with c2: st.metric("💰 Con valor", f"{df['market_value_in_eur'].notna().sum():,}")
with c3: st.metric("💎 Chollos", f"{len(df_chollos):,}")
with c4: st.metric("📈 R² XGBoost", "0.73")
st.divider()

# ─── TABS ───
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Scout IA","💎 Detector de Chollos","🗺️ Mapa Táctico","📊 EDA"])

# ══ TAB 1: SCOUT IA ══
with tab1:
    st.markdown("### 🔍 Búsqueda de Jugadores por Perfil Natural")
    st.caption("El agente RAG busca jugadores similares en la base de +2500 jugadores.")
    cq, cp = st.columns([3,1])
    with cq:
        query_text = st.text_area("Describe el perfil:",
            placeholder='Ej: "pivote defensivo similar a Busquets, max €8M"', height=100)
    with cp:
        pos_scout = st.selectbox("Posición (opc.)", ["Auto", "POR", "DFC", "LI/LD", "MC", "MCO", "DC"])
        n_results = st.slider("N° candidatos", 3, 10, 5)
        only_bargains = st.checkbox("Solo chollos (ratio > 2x)")
    if st.button("🔍 Buscar Candidatos", type="primary", use_container_width=True) and query_text:
        with st.spinner("Consultando ChromaDB + Gemini LLM..."):
            try:
                collection, encoder, shap_df, scout_exec, extractor = load_agents()
                pos = None if pos_scout=="Auto" else FC_POS_MAP_INV.get(pos_scout, pos_scout)
                
                # Optimización: Extracción automática de posición y liga con Gemini
                extracted_leagues = None
                try:
                    prompt = f"""Analiza esta consulta de scouting: '{query_text}'.
Extrae la posición y las ligas mencionadas.
Posiciones: MF, FW, CB, FB, GK, AM.
Ligas (códigos): GB1 (Inglaterra), ES1 (España), IT1 (Italia), L1 (Alemania), FR1 (Francia), BRA1 (Brasil), ARG1 (Argentina), NL1 (Holanda), PO1 (Portugal). "5 grandes ligas" = ["GB1", "ES1", "IT1", "L1", "FR1"].
Responde ÚNICAMENTE con un JSON en este formato:
{{"position": "MF", "leagues": ["GB1", "ES1"]}}
Usa null si no especifica. Nada de markdown."""
                    resp = extractor.invoke([HumanMessage(content=prompt)]).content.strip()
                    resp = resp.replace('```json','').replace('```','').strip()
                    extracted_data = json.loads(resp)
                    
                    if pos is None and extracted_data.get("position"):
                        pos = extracted_data["position"]
                    if extracted_data.get("leagues"):
                        extracted_leagues = extracted_data["leagues"]
                except Exception as e:
                    pass

                final_leagues = extracted_leagues if extracted_leagues else league_filter
                max_p = max_price if only_bargains else None
                candidates = query_similar_players(query_text, collection, encoder,
                                                    n_results, pos, max_p, leagues=final_leagues)
                if only_bargains:
                    candidates = [c for c in candidates if c.get("value_ratio",1)>2.0]
                
                # Extraer SHAP para los candidatos: O(1) lookup optimizado con to_dict en C
                shap_data = {}
                player_to_idx = df_valued.reset_index().set_index("player")["index"].to_dict()
                for c in candidates:
                    pname = c["player_name"]
                    if pname in player_to_idx:
                        idx = player_to_idx[pname]
                        if idx in shap_df.index:
                            s_row = shap_df.loc[idx]
                            top_features = s_row.abs().nlargest(3).index
                            shap_data[pname] = {f: ("+" if s_row[f]>0 else "-") for f in top_features}

                llm_report = generate_scout_report(candidates, query_text, shap_data)
                st.markdown("#### 📋 Candidatos Encontrados")
                for i, c in enumerate(candidates):
                    icon = "💎" if c.get("value_ratio",1)>2 else "⚽"
                    with st.expander(f"{icon} {c['player_name']} — {c['team']} ({c['league']}) | Sim: {c['similarity_score']:.2%}", expanded=(i==0)):
                        ca,cb,cc = st.columns(3)
                        with ca:
                            st.metric("Valor Actual", f"€{c['market_value_eur']/1e6:.1f}M")
                            st.metric("Valor Predicho", f"€{c['predicted_value_eur']/1e6:.1f}M")
                        with cb:
                            st.metric("Posición", FC_POS_MAP.get(c["position"], c["position"]))
                            st.metric("Estilo", c.get("cluster_label","N/A"))
                        with cc:
                            r = c.get("value_ratio",1)
                            color = "🟢" if r>2 else ("🔴" if r<0.5 else "🟡")
                            st.metric(f"{color} Ratio", f"{r:.2f}x")
                        stats = c.get("stats",{})
                        sdf = pd.DataFrame({"Métrica":["PrgP/90","xAG/90","xG/90","Presión%","Tkl+Int/90"],
                            "Valor":[stats.get("prgp_per90",0),stats.get("xag_per90",0),
                                     stats.get("xg_per90",0),stats.get("press_pct",0),
                                     stats.get("tkl_int_per90",0)]})
                        st.bar_chart(sdf.set_index("Métrica"))
                st.markdown("#### 🤖 Análisis del Agente IA")
                st.markdown(llm_report)
            except Exception as e:
                st.error(f"Error: {e}")
                st.info("Asegúrate de tener GOOGLE_API_KEY configurada y los datos procesados.")

# ══ TAB 2: DETECTOR DE CHOLLOS ══
with tab2:
    st.markdown("### 💎 Detector de Chollos en el Mercado")
    dv = df_valued.copy()
    if position_filter!="Todas": dv = dv[dv.get("position_group","")==position_filter]
    if league_filter and "league" in dv.columns: dv = dv[dv["league"].isin(league_filter)]
    dv = dv[dv["market_value_in_eur"]<=max_price]
    dv = dv.dropna(subset=["market_value_in_eur","predicted_value_eur"])
    if len(dv)>0 and "value_ratio" in dv.columns:
        dv["categoría"] = dv["value_ratio"].apply(lambda r: "💎 Chollo" if r>2 else ("⚠️ Sobrevalorado" if r<0.5 else "✅ Precio Justo"))
        fig = px.scatter(dv, x=dv["market_value_in_eur"]/1e6, y=dv["predicted_value_eur"]/1e6,
                         color="categoría", hover_data=["player","squad","position_group"],
                         color_discrete_map={"💎 Chollo":"#27ae60","✅ Precio Justo":"#3498db","⚠️ Sobrevalorado":"#e74c3c"},
                         title="Valor Real vs. Predicho (M€)")
        mx = max(dv["market_value_in_eur"].max(),dv["predicted_value_eur"].max())/1e6
        fig.add_trace(go.Scatter(x=[0,mx],y=[0,mx],mode="lines",line=dict(color="gray",dash="dash",width=1),name="y=x"))
        fig.update_layout(height=600, plot_bgcolor="white")
        st.plotly_chart(fig, width="stretch")
    st.markdown("#### 🏆 Top 10 Chollos")
    cf = df_chollos.copy()
    if position_filter!="Todas" and "position_group" in cf.columns:
        cf = cf[cf["position_group"]==position_filter]
    if "market_value_in_eur" in cf.columns:
        cf = cf[cf["market_value_in_eur"]<=max_price]
    for _, row in cf.head(10).iterrows():
        with st.expander(f"💎 {row['player']} — {row.get('squad','')} | €{row['market_value_in_eur']/1e6:.1f}M → €{row['predicted_value_eur']/1e6:.1f}M ({row['value_ratio']:.1f}x)"):
            ca,cb = st.columns(2)
            with ca: st.metric("Valor Actual",f"€{row['market_value_in_eur']/1e6:.1f}M"); st.metric("Predicho",f"€{row['predicted_value_eur']/1e6:.1f}M")
            with cb: st.metric("Ratio",f"{row['value_ratio']:.2f}x"); st.metric("Posición",row.get("position_group","N/A"))
            if st.button("🤖 Analizar con IA", key=f"llm_{row['player']}"):
                with st.spinner("Generando análisis..."):
                    st.markdown(generate_bargain_report(row.to_dict()))

# ══ TAB 3: MAPA TÁCTICO ══
with tab3:
    st.markdown("### 🗺️ Mapa Táctico de Jugadores (UMAP 2D)")
    pos_umap = st.selectbox("Posición:", ["MC", "DC", "DFC", "LI/LD"], key="umap_pos")
    du = df[(df["position_group"]==pos_umap)&df["umap_x"].notna()&df["umap_y"].notna()].copy() if "umap_x" in df.columns else pd.DataFrame()
    if len(du)>0:
        fig = px.scatter(du, x="umap_x", y="umap_y", color="cluster_label",
                         hover_data={"player":True,"squad":True,"league":True,
                                     "market_value_in_eur":":.0f","umap_x":False,"umap_y":False},
                         title=f"Clusters Tácticos — {pos_umap} ({len(du)} jugadores)",
                         color_discrete_sequence=px.colors.qualitative.Set2, height=600)
        fig.update_layout(plot_bgcolor="white",
                          xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
                          yaxis=dict(showgrid=False,zeroline=False,showticklabels=False))
        st.plotly_chart(fig, width="stretch")
        cs = du.groupby("cluster_label").agg(Jugadores=("player","count"),
            Valor_medio=("market_value_in_eur", lambda x: f"€{x.mean()/1e6:.1f}M")).reset_index()
        st.dataframe(cs, use_container_width=True)
    else:
        st.warning(f"No hay datos UMAP para {pos_umap}. Ejecuta el ML Agent primero.")

# ══ TAB 4: EDA ══
with tab4:
    st.markdown("### 📊 Análisis Exploratorio de Datos")
    e1,e2 = st.columns(2)
    with e1:
        fig = px.histogram(df.dropna(subset=["market_value_in_eur"]), x="market_value_in_eur",
                           nbins=50, log_x=True, color="position_group",
                           title="Distribución Valor de Mercado (log)")
        st.plotly_chart(fig, width="stretch")
    with e2:
        if "league" in df.columns:
            fig = px.box(df.dropna(subset=["market_value_in_eur","league"]),
                         x="league", y="market_value_in_eur", color="league",
                         title="Valor por Liga", log_y=True)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width="stretch")
    st.markdown("#### 📋 Dataset Completo")
    show_cols = ["player","squad","league","position_group","cluster_label",
                 "market_value_in_eur","predicted_value_eur","value_ratio"]
    avail = [c for c in show_cols if c in df.columns]
    st.dataframe(df[avail].sort_values("value_ratio",ascending=False,na_position="last"),
                 use_container_width=True, height=400)
