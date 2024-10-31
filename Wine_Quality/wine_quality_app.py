import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Sett page config
st.set_page_config(page_title="Vin Kvalitetsvurdering", page_icon="🍷")

# Last inn modellen
try:
    model = joblib.load('wine_quality_model.pkl')
except:
    st.error("Kunne ikke laste modellen. Sjekk at 'wine_quality_model.pkl' eksisterer.")
    st.stop()

st.title("🍷🍷Vin Kvalitetsvurdering🍷🍷")
st.markdown("### Fyll inn vinens egenskaper")

# Lag input felter
col1, col2 = st.columns(2)

with col1:
    fixed_acidity = st.number_input(
        'Fixed Acidity (g/L)',
        min_value=4.0,
        max_value=16.0,
        value=7.0,
        step=0.1
    )
    
    volatile_acidity = st.number_input(
        'Volatile Acidity (g/L)',
        min_value=0.1,
        max_value=1.2,
        value=0.27,
        step=0.01
    )
    
    citric_acid = st.number_input(
        'Citric Acid (g/L)',
        min_value=0.0,
        max_value=1.0,
        value=0.36,
        step=0.01
    )
    
    residual_sugar = st.number_input(
        'Residual Sugar (g/L)',
        min_value=0.9,
        max_value=30.0,
        value=2.0,
        step=0.1
    )
    
    chlorides = st.number_input(
        'Chlorides (g/L)',
        min_value=0.012,
        max_value=0.611,
        value=0.045,
        step=0.001,
        format="%.3f"
    )

with col2:
    free_sulfur_dioxide = st.number_input(
        'Free Sulfur Dioxide (mg/L)',
        min_value=1,
        max_value=72,
        value=30,
        step=1
    )
    
    total_sulfur_dioxide = st.number_input(
        'Total Sulfur Dioxide (mg/L)',
        min_value=6,
        max_value=289,
        value=100,
        step=1
    )
    
    density = st.number_input(
        'Density (g/cm³)',
        min_value=0.99007,
        max_value=1.00369,
        value=0.9967,
        step=0.00001,
        format="%.5f"
    )
    
    ph = st.number_input(
        'pH',
        min_value=2.74,
        max_value=4.01,
        value=3.2,
        step=0.01
    )
    
    sulphates = st.number_input(
        'Sulphates (g/L)',
        min_value=0.33,
        max_value=2.0,
        value=0.45,
        step=0.01
    )
    
    alcohol = st.number_input(
        'Alcohol (%)',
        min_value=8.4,
        max_value=14.9,
        value=11.0,
        step=0.1
    )

def vurder_kvalitet(features):
    """Analyserer hver komponent og gir spesifikk tilbakemelding"""
    [fixed_acidity, volatile_acidity, citric_acid, residual_sugar, 
     chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, 
     ph, sulphates, alcohol] = features[0]
    
    analyse = {
        "Syrebalanse": {
            "status": "god" if (6.5 <= fixed_acidity <= 8.5 and 0.2 <= volatile_acidity <= 0.4) else "trenger justering",
            "forklaring": "Syrebalansen påvirker vinens friskhet og holdbarhet",
            "verdi": f"Fast syre: {fixed_acidity:.1f}, Flyktig syre: {volatile_acidity:.2f}"
        },
        "Sukkerinnhold": {
            "status": "balansert" if (2 <= residual_sugar <= 15) else "ubalansert",
            "forklaring": "Påvirker vinens sødme og kropp",
            "verdi": f"{residual_sugar:.1f} g/L"
        },
        "Alkoholnivå": {
            "status": "optimal" if (11 <= alcohol <= 14) else "ikke optimal",
            "forklaring": "Bidrar til vinens kropp og kompleksitet",
            "verdi": f"{alcohol:.1f}%"
        },
        "Konservering": {
            "status": "god" if (30 <= free_sulfur_dioxide <= 60) else "trenger justering",
            "forklaring": "Viktig for vinens stabilitet og lagring",
            "verdi": f"Fri SO2: {free_sulfur_dioxide:.0f} mg/L"
        },
        "pH-verdi": {
            "status": "optimal" if (3.0 <= ph <= 3.4) else "ikke optimal",
            "forklaring": "Påvirker smak og mikrobiologisk stabilitet",
            "verdi": f"{ph:.2f}"
        }
    }
    return analyse

# Analyser-knapp og resultatvisning
if st.button('Analyser vin', type='primary'):
    inputs = [fixed_acidity, volatile_acidity, citric_acid, residual_sugar, 
              chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, 
              ph, sulphates, alcohol]
    
    features = np.array([inputs])
    
    try:
        quality = model.predict(features)
        
        # Vis total score
        st.markdown("## Kvalitetsanalyse")
        st.metric("Total kvalitetsscore", f"{quality[0]:.1f}/10")
        
        # Vis detaljert analyse
        analyse = vurder_kvalitet(features)
        
        st.markdown("### Detaljert analyse av komponenter")
        
        for komponent, info in analyse.items():
            with st.expander(f"{komponent} - {info['status']}"):
                st.write(f"**Målt verdi:** {info['verdi']}")
                st.write(f"**Betydning:** {info['forklaring']}")
                
                if info['status'] == "god" or info['status'] == "optimal" or info['status'] == "balansert":
                    st.success(f"Status: {info['status']}")
                else:
                    st.warning(f"Status: {info['status']}")
        
        # Vis kvalitetsskala forklaring
        with st.expander("ℹ️ Hva betyr kvalitetsskalaen?"):
            st.markdown("""
            **Vinkvalitet (0-10) vurderes basert på følgende kriterier:**
            
            #### 7-10: Utmerket kvalitet
            - Perfekt balanse mellom alle komponenter
            - Optimal syrebalanse
            - Ideelt alkoholnivå
            - Korrekt konserveringsnivå
            - Stabil pH-verdi
            
            #### 5-6: God kvalitet
            - God balanse, men med rom for forbedring
            - Akseptable nivåer av alle komponenter
            - Sikker for konsum
            - Typisk for kommersielle viner
            
            #### Under 5: Trenger forbedring
            - Ubalanse i en eller flere komponenter
            - Mulige problemer med syreinnhold, alkoholnivå eller pH
            """)
        
        # Vis anbefalinger
        st.markdown("### Anbefalinger for forbedring")
        anbefalinger = []
        for komponent, info in analyse.items():
            if info['status'] != "god" and info['status'] != "optimal" and info['status'] != "balansert":
                anbefalinger.append(f"- Juster {komponent.lower()} for bedre balanse")
        
        if anbefalinger:
            st.write("\n".join(anbefalinger))
        else:
            st.success("Alle komponenter er innenfor optimale verdier!")
            
    except Exception as e:
        st.error(f"En feil oppstod under analyse: {str(e)}")