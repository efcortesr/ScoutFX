# 📖 Guía de Usuario — ScoutFX

## Requisitos Previos
- Python 3.10+
- Cuenta gratuita en [Google AI Studio](https://aistudio.google.com) para obtener API key
- Dataset de [Transfermarkt en Kaggle](https://www.kaggle.com/datasets/davidcariboo/player-scores)

## Instalación Paso a Paso

### 1. Clonar repositorio e instalar dependencias
```bash
git clone https://github.com/TU_USUARIO/football-iq-eafit.git
cd football-iq-eafit
pip install -r requirements.txt
```

### 2. Configurar API Key
```bash
cp .env.example .env
# Editar .env y pegar tu GOOGLE_API_KEY
```

### 3. Preparar datos
- Descargar `players.csv` de Kaggle → `data/raw/players.csv`
- Ejecutar notebooks en orden (01 → 04)

### 4. Correr la app
```bash
streamlit run app/main.py
```

## Uso de la Aplicación

### Tab 1: Scout IA 🔍
Escribe en lenguaje natural el perfil que buscas:
- "pivote defensivo similar a Busquets, max €8M"
- "delantero goleador de la Bundesliga"

### Tab 2: Detector de Chollos 💎
- Visualiza el scatter plot de valor real vs predicho
- Los puntos verdes son chollos detectados por XGBoost
- Haz clic en "Analizar con IA" para obtener justificación del LLM

### Tab 3: Mapa Táctico 🗺️
- Selecciona una posición (MF, FW, CB, FB)
- Explora los clusters tácticos en el mapa UMAP 2D
- Jugadores cercanos tienen estilos similares

### Tab 4: EDA 📊
- Explora distribuciones de valor de mercado
- Filtra por liga, posición y precio
- Exporta datos para análisis adicional
