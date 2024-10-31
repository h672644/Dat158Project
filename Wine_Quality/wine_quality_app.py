import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler

import pickle


# Sett page config med wide layout for bedre visualiseringer
st.set_page_config(
    page_title="Avansert Vin Kvalitetsvurdering",
    page_icon="🍷",
    layout="wide"
)

# Custom CSS for bedre utseende
st.markdown("""
    <style>
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .st-emotion-cache-1p11qjc {
        padding: 2rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Definer ideelle verdiområder
ideal_ranges = {
    "fixed_acidity": {"min": 6.0, "max": 9.0, "ideal": 7.5, "unit": "g/L"},
    "volatile_acidity": {"min": 0.2, "max": 0.4, "ideal": 0.3, "unit": "g/L"},
    "citric_acid": {"min": 0.25, "max": 0.5, "ideal": 0.35, "unit": "g/L"},
    "residual_sugar": {"min": 1.5, "max": 5.0, "ideal": 2.5, "unit": "g/L"},
    "chlorides": {"min": 0.04, "max": 0.1, "ideal": 0.06, "unit": "g/L"},
    "free_sulfur_dioxide": {"min": 20, "max": 40, "ideal": 30, "unit": "mg/L"},
    "total_sulfur_dioxide": {"min": 80, "max": 160, "ideal": 120, "unit": "mg/L"},
    "density": {"min": 0.995, "max": 0.998, "ideal": 0.996, "unit": "g/cm³"},
    "ph": {"min": 3.2, "max": 3.6, "ideal": 3.4, "unit": ""},
    "sulphates": {"min": 0.5, "max": 0.8, "ideal": 0.65, "unit": "g/L"},
    "alcohol": {"min": 11.0, "max": 13.0, "ideal": 12.0, "unit": "%"}
}

model = pickle.load(open("wine_quality_model.pkl", "rb"))

# Last inn modellen med caching
@st.cache_resource
def load_model():
    try:
        return joblib.load('wine_quality_model.pkl')
    except Exception as e:
        st.error(f"Kunne ikke laste modellen: {str(e)}")
        st.stop()

model = load_model()

# Header
st.title("🍷 Avansert Vin Kvalitetsvurdering")
st.markdown("### En detaljert analyse av vinens egenskaper")

# Opprett tabs for ulike input-metoder
tab1, tab2 = st.tabs(["📊 Slider Input", "📝 Numerisk Input"])

# Dictionary for å lagre input-verdier
inputs = {}

with tab1:
    # Opprett to kolonner for input fields
    slider_col1, slider_col2 = st.columns(2)
    
    for i, (key, range_info) in enumerate(ideal_ranges.items()):
        with slider_col1 if i < len(ideal_ranges)/2 else slider_col2:
            inputs[key] = st.slider(
                f"{key.replace('_', ' ').title()} ({range_info['unit']})",
                min_value=float(range_info['min']),
                max_value=float(range_info['max']),
                value=float(range_info['ideal']),
                help=f"Ideelt område: {range_info['min']}-{range_info['max']} {range_info['unit']}"
            )

with tab2:
    # Opprett to kolonner for input fields
    num_col1, num_col2 = st.columns(2)
    
    for i, (key, range_info) in enumerate(ideal_ranges.items()):
        with num_col1 if i < len(ideal_ranges)/2 else num_col2:
            inputs[key] = st.number_input(
                f"{key.replace('_', ' ').title()} ({range_info['unit']})",
                min_value=float(range_info['min']),
                max_value=float(range_info['max']),
                value=float(range_info['ideal']),
                help=f"Ideelt område: {range_info['min']}-{range_info['max']} {range_info['unit']}"
            )

# Analyse-knapp
if st.button('🔍 Analyser Vin', use_container_width=True):
    try:
        # Forbered input data
        input_list = [inputs[key] for key in ideal_ranges.keys()]
        features = np.array([input_list])
        
        # Gjør prediksjon
        quality = model.predict(features)
        
        # Opprett kolonner for resultater og visualiseringer
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            # Valider prediksjon
            if quality[0] <= 0 or quality[0] > 10:
                st.error("""
                🚫 Ugyldig prediksjon detektert
                
                Dette kan skyldes:
                1. Uvanlig kombinasjon av verdier
                2. Modellen trenger rekalibrering
                
                Prøv å justere verdiene nærmere ideelle områder.
                """)
            else:
                # Vis kvalitetsscore
                st.metric(
                    "Vinkvalitet",
                    f"{quality[0]:.1f}/10",
                    delta=f"{quality[0]-5:.1f} fra middels",
                    delta_color="normal"
                )
                
                # Vis kvalitetsvurdering
                if quality[0] >= 7:
                    st.success("🌟 Utmerket vinkvalitet!")
                elif quality[0] >= 5:
                    st.info("✨ God vinkvalitet")
                else:
                    st.warning("""
                    ⚠️ Vinen kan forbedres
                    
                    Mulige årsaker:
                    - Ubalanse i syre-nivåer
                    - Suboptimale fermenteringsforhold
                    - Behov for justering av sulfat-nivåer
                    
                    Prøv å justere verdiene nærmere ideelle områder.
                    """)
                
                # Vis input-verdiene
                st.markdown("### Dine input-verdier:")
                input_df = pd.DataFrame([inputs], columns=ideal_ranges.keys())
                st.dataframe(input_df)
        
        with res_col2:
            # Radarplot for sammenligning med ideelle verdier
            categories = list(ideal_ranges.keys())
            current_values = [inputs[key] for key in categories]
            ideal_values = [ideal_ranges[key]["ideal"] for key in categories]
            
            # Normaliser verdiene
            scaler = MinMaxScaler()
            normalized_current = scaler.fit_transform(np.array(current_values).reshape(-1, 1)).flatten()
            normalized_ideal = scaler.transform(np.array(ideal_values).reshape(-1, 1)).flatten()
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=normalized_current,
                theta=categories,
                fill='toself',
                name='Dine verdier'
            ))
            
            fig.add_trace(go.Scatterpolar(
                r=normalized_ideal,
                theta=categories,
                fill='toself',
                name='Ideelle verdier'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title="Sammenligning med ideelle verdier"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Vis avviksanalyse
        st.markdown("### 📊 Avviksanalyse")
        avvik_col1, avvik_col2 = st.columns(2)
        
        with avvik_col1:
            # Analyser og vis betydelige avvik
            avvik = []
            for key, value in inputs.items():
                ideal = ideal_ranges[key]["ideal"]
                if abs(value - ideal) > 0.1 * (ideal_ranges[key]["max"] - ideal_ranges[key]["min"]):
                    avvik.append(f"- {key.replace('_', ' ').title()}: {value:.2f} (ideal: {ideal:.2f})")
            
            if avvik:
                st.warning("Betydelige avvik fra ideelle verdier:")
                for a in avvik:
                    st.markdown(a)
            else:
                st.success("Ingen betydelige avvik fra ideelle verdier! 👍")
        
        with avvik_col2:
            # Stolpediagram for avvik fra ideelle verdier
            avvik_data = []
            for key in ideal_ranges.keys():
                ideal = ideal_ranges[key]["ideal"]
                current = inputs[key]
                avvik_prosent = ((current - ideal) / ideal) * 100
                avvik_data.append({
                    "Parameter": key.replace('_', ' ').title(),
                    "Avvik (%)": avvik_prosent
                })
            
            avvik_df = pd.DataFrame(avvik_data)
            fig = px.bar(
                avvik_df,
                x="Parameter",
                y="Avvik (%)",
                title="Avvik fra ideelle verdier",
                color="Avvik (%)",
                color_continuous_scale=["red", "yellow", "green"]
            )
            
            fig.update_layout(
                xaxis_tickangle=-45,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"""
        En feil oppstod under analyse: {str(e)}
        
        Dette kan skyldes:
        1. Uventede verdier i input
        2. Problem med modellen
        3. Manglende eller korrupt modell-fil
        
        Kontakt support hvis problemet vedvarer.
        """)

# Informasjonsboks
with st.expander("ℹ️ Om vinanalysen"):
    st.markdown("""
    ### Hvordan tolke resultatene:
    
    - **Vinkvalitet**: Score fra 0-10 hvor høyere er bedre
    - **Ideelle områder**: Verdier basert på analyser av kvalitetsviner
    - **Avviksanalyse**: Identifiserer verdier som avviker betydelig fra idealet
    
    ### Tips for forbedring:
    
    1. **Syre-balanse**:
       - Hold pH mellom 3.2 og 3.6
       - Juster fixed acidity innenfor 6.0-9.0 g/L
    
    2. **Sulfitter**:
       - Free SO2 under 40 mg/L
       - Total SO2 under 160 mg/L
    
    3. **Alkohol og sukker**:
       - Alkoholinnhold mellom 11-13%
       - Residual sugar under 5.0 g/L for tørre viner
    """)