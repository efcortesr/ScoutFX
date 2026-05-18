# ⚽ ScoutFX — Scout Inteligente + Detector de Chollos en Transferencias

> Proyecto Final · Inteligencia Artificial · Universidad EAFIT 2026-1

Sistema multiagente que combina **clustering táctico**, **valuación de mercado con XGBoost**
y un **agente RAG con LLM** para identificar talento subvalorado en las 5 grandes ligas europeas.

---

## 🧩 ¿Qué hace el sistema?

| Módulo | Descripción |
|--------|-------------|
| 🔍 **Scout IA** | Búsqueda en lenguaje natural → "dame un pivote defensivo similar a Busquets, max €10M" |
| 💎 **Detector de Chollos** | XGBoost predice el valor justo; si el predicho > 2x precio actual = chollo |
| 🗺️ **Mapa Táctico** | UMAP 2D agrupa jugadores por estilo de juego (clusters tácticos) |
| 🤖 **Analista LLM** | Gemini 2.5 Flash justifica cada recomendación con stats reales y SHAP values |

---

## 🛠️ Instalación

**Requisitos:** Python 3.10+

```bash
git clone https://github.com/TU_USUARIO/football-iq-eafit.git
cd football-iq-eafit
pip install -r requirements.txt
```

---

## 🔑 Configuración de API Key

Crear archivo `.env` en la raíz del proyecto:

```
GOOGLE_API_KEY=xxxxxxxxxxxxxxxxxxxxxx
```

Obtener gratis en: https://aistudio.google.com/apikey

---

## 📊 Preparación de Datos

### Paso 1 — Descargar dataset Transfermarkt

1. Ir a https://www.kaggle.com/datasets/davidcariboo/player-scores
2. Descargar `players.csv`
3. Guardar en `data/raw/players.csv`


---

## 🚀 Correr la Aplicación

```bash
streamlit run app/main.py
```

Abrir: **http://localhost:8501**

---

## 📁 Estructura del Proyecto

```
football-iq-eafit/
├── README.md
├── requirements.txt
├── .env                          
├── .gitignore
├── docs/
│   ├── informe_final.pdf
│   └── guia_usuario.md
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_clustering.ipynb
│   ├── 03_market_value.ipynb
│   └── 04_rag_agent.ipynb
├── src/
│   ├── data/
│   │   ├── scraper_fbref.py
│   │   ├── loader_transfermarkt.py
│   │   └── preprocessor.py
│   ├── models/
│   │   ├── clustering.py
│   │   └── market_value.py
│   ├── agents/
│   │   ├── embedding_agent.py
│   │   ├── scout_rag_agent.py
│   │   └── llm_analyst.py
│   └── evaluation/
│       ├── eval_clustering.py
│       └── eval_rag.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── checkpoints/
└── app/
    └── main.py
```

---

## 📈 Resultados del Modelo

| Modelo | Métrica | Valor |
|--------|---------|-------|
| KMeans (MF) | Silhouette Score | 0.31 |
| KMeans (FW) | Silhouette Score | 0.28 |
| XGBoost | R² (escala log) | 0.73 |
| XGBoost | RMSE | €6.2M |
| RAG Eval | Accuracy (20 queries) | 75% |

---

## 🎥 Video Demo

📹 **Link:** [https://eafit-my.sharepoint.com/:v:/g/personal/efcortesr_eafit_edu_co/IQDF9MVmQJ8fSLI3fpZABGkSAaXB2dOYkdVSlJh_G-F3xHg?e=yOkZgr]

---

