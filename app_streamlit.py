import streamlit as st
import librosa
import numpy as np
import joblib
import os
import pandas as pd
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import matplotlib
import json
matplotlib.use('Agg')  # Usar backend sin GUI
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Clasificador de Cetáceos",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuración de sesión
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'show_advanced_analysis' not in st.session_state:
    st.session_state.show_advanced_analysis = False
if 'show_model_info' not in st.session_state:
    st.session_state.show_model_info = False

 # Función para cargar datos de frecuencias reales
def load_real_frequency_data():
    """Carga los datos de frecuencias reales desde el archivo JSON generado por el notebook"""
    try:
        json_path = "/home/azureuser/cloudfiles/code/frequency_ranges.json"
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            print("⚠️ Archivo frequency_ranges.json no encontrado, usando datos genéricos")
            return None
    except Exception as e:
        print(f"⚠️ Error cargando datos de frecuencias: {e}")
        return None

# Cargar datos reales de frecuencias
REAL_FREQUENCY_DATA = load_real_frequency_data()

# Información sobre los cetáceos
CETACEAN_INFO = {
    "blue-whale-balaenoptera-musculus": {
        "nombre_comun": "Ballena Azul",
        "nombre_cientifico": "Balaenoptera musculus",
        "descripcion": "La ballena azul es el animal más grande que ha existido jamás en la Tierra. Puede alcanzar hasta 30 metros de largo y pesar hasta 190 toneladas. Es conocida por sus profundos cantos submarinos que pueden escucharse a kilómetros de distancia. Se alimenta principalmente de krill (pequeños crustáceos) y es una especie en peligro de extinción.",
        "color": "#1976d2",
        "familia": "Balaenopteridae",
        "tipo_voz": "Cantos profundos y resonantes",
        "rango_frecuencia": "Muy baja (10-188 Hz)"
    },
    "fin-whale-balaenoptera physalus": {
        "nombre_comun": "Ballena de Aleta",
        "nombre_cientifico": "Balaenoptera physalus",
        "descripcion": "La ballena de aleta es la segunda especie más grande de ballena. Puede alcanzar hasta 24 metros de largo. Es conocida por su asimetría de coloración (el lado derecho es más claro que el izquierdo). Es una nadadora rápida y realiza migraciones anuales entre aguas polares y tropicales.",
        "color": "#b0bec5",
        "familia": "Balaenopteridae",
        "tipo_voz": "Llamadas repetitivas y pulsantes",
        "rango_frecuencia": "Baja (20-100 Hz)"
    },
    "humpback-whale-megaptera-novaeangliae": {
        "nombre_comun": "Ballena Jorobada",
        "nombre_cientifico": "Megaptera novaeangliae",
        "descripcion": "La ballena jorobada es famosa por sus acrobacias acuáticas y sus complejos cantos. Los machos cantan canciones elaboradas que pueden durar horas. Es conocida por sus aletas largas (hasta 5 metros) y sus saltos espectaculares fuera del agua. Estas ballenas son muy activas y sociales.",
        "color": "#8d6e63",
        "familia": "Megapteridae",
        "tipo_voz": "Complejos cantos y melodías",
        "rango_frecuencia": "Muy baja (30-8000 Hz)"
    },
    "killer-whale-orcinus-orca": {
        "nombre_comun": "Orca",
        "nombre_cientifico": "Orcinus orca",
        "descripcion": "A menudo llamada 'ballena asesina', la orca es en realidad el delfín más grande. Son depredadores ápice inteligentes y sociales que viven en grupos familiares llamados vainas. Son conocidas por sus acrobacias y su comportamiento de caza coordinado. Tienen una aleta dorsal distintiva.",
        "color": "#212121",
        "familia": "Delphinidae",
        "tipo_voz": "Clics, silbidos y pulsaciones",
        "rango_frecuencia": "Alta (500-20000 Hz)"
    },
    "minke-whale-balaenoptera-acutorostrata": {
        "nombre_comun": "Ballena Enana",
        "nombre_cientifico": "Balaenoptera acutorostrata",
        "descripcion": "La ballena enana es la más pequeña de las ballenas barbadas, alcanzando solo 9-10 metros de largo. A pesar de su tamaño pequeño, es muy ágil y rápida. A menudo se acerca a embarcaciones por curiosidad. Se distribuye en aguas frías y subpolares.",
        "color": "#1565c0",
        "familia": "Balaenopteridae",
        "tipo_voz": "Llamadas de pulso y tonos",
        "rango_frecuencia": "Baja (50-1000 Hz)"
    },
    "north-atlantic-right-whale-eubalaena-glacialis": {
        "nombre_comun": "Ballena Franca del Atlántico Norte",
        "nombre_cientifico": "Eubalaena glacialis",
        "descripcion": "La ballena franca del Atlántico Norte es una de las especies más en peligro de extinción, con menos de 350 individuos restantes. Es fácilmente identificable por los callosidades (verrugas) en su cabeza. Se alimenta filtrando el agua y vive principalmente en el Océano Atlántico Norte.",
        "color": "#6d4c41",
        "familia": "Balaenidae",
        "tipo_voz": "Gruñidos y llamadas sociales",
        "rango_frecuencia": "Baja (50-500 Hz)"
    },
    "northern-bottlenose-whale-hyperoodon-ampullatus": {
        "nombre_comun": "Ballena Pico del Norte",
        "nombre_cientifico": "Hyperoodon ampullatus",
        "descripcion": "La ballena pico del norte es un cetáceo de tamaño mediano (7-10 metros) que vive en aguas profundas del Atlántico Norte. Pueden sumergirse a profundidades extremas (más de 2000 metros) para alimentarse de calamares. Emiten clics y silbidos para la ecolocalización.",
        "color": "#2196f3",
        "familia": "Ziphiidae",
        "tipo_voz": "Clics y ecolocalización",
        "rango_frecuencia": "Media-Alta (2000-130000 Hz)"
    },
    "pilot-whale-globicephala-spp": {
        "nombre_comun": "Ballena Piloto",
        "nombre_cientifico": "Globicephala spp",
        "descripcion": "Las ballenas piloto son delfines grandes que viven en grupos sociales muy unidos llamadas vainas. Son conocidas por su lealtad grupal extrema. Tienen cuerpos robustos de color gris oscuro a negro. Se alimentan principalmente de calamares en aguas profundas.",
        "color": "#1976d2",
        "familia": "Delphinidae",
        "tipo_voz": "Clics, silbidos y llamadas sociales",
        "rango_frecuencia": "Media-Alta (300-130000 Hz)"
    },
    "sei-whale-balaenoptera-borealis": {
        "nombre_comun": "Ballena Sei",
        "nombre_cientifico": "Balaenoptera borealis",
        "descripcion": "La ballena sei es una ballena barbada de tamaño medio (14-18 metros) que es extremadamente rápida, alcanzando velocidades de hasta 50 km/h. Es difícil de ver en la naturaleza porque raramente sale completamente del agua. Se distribuye en todos los océanos del mundo.",
        "color": "#00bcd4",
        "familia": "Balaenopteridae",
        "tipo_voz": "Llamadas y pulsaciones",
        "rango_frecuencia": "Baja (20-400 Hz)"
    },
    "sperm-whale-physeter-macrocephalus": {
        "nombre_comun": "Cachalote",
        "nombre_cientifico": "Physeter macrocephalus",
        "descripcion": "El cachalote es el cetáceo dentado más grande del mundo. Tiene la cabeza más grande de cualquier animal viviente. Puede sumergirse a profundidades extraordinarias (hasta 3000 metros) para cazar calamares gigantes. Su cerebro es el más grande del reino animal.",
        "color": "#8d6e63",
        "familia": "Physeteridae",
        "tipo_voz": "Clics rítmicos (codas)",
        "rango_frecuencia": "Media-Alta (400-30000 Hz)"
    }
}

# Actualizar rangos de frecuencia con datos reales si están disponibles
if REAL_FREQUENCY_DATA:
    for species, real_data in REAL_FREQUENCY_DATA.items():
        if species in CETACEAN_INFO:
            # Actualizar rango de frecuencia con datos reales
            min_hz = real_data.get('rango_min_hz', 0)
            max_hz = real_data.get('rango_max_hz', 0)
            if min_hz > 0 and max_hz > 0:
                CETACEAN_INFO[species]['rango_frecuencia_real'] = f"{min_hz:.1f}-{max_hz:.1f} Hz"
                CETACEAN_INFO[species]['frecuencia_media_real'] = real_data.get('frecuencia_media_hz', 0)
                CETACEAN_INFO[species]['frecuencia_pico_real'] = real_data.get('frecuencia_pico_hz', 0)
                print(f"✓ Actualizado {species}: {min_hz:.1f}-{max_hz:.1f} Hz")
    print("✅ Rangos de frecuencia actualizados con datos reales del espectrograma")
else:
    print("⚠️ Usando rangos de frecuencia genéricos (no se encontraron datos reales)")

# Información sobre familias de cetáceos
FAMILY_INFO = {
    "Balaenopteridae": {
        "nombre": "Ballenas Barbadas (Rorcuales)",
        "descripcion": "Familipción de ballenas barbadas grandes. Incluyen las especies más grandes del planeta. Se alimentan filtrando agua. Generalmente emiten cantos profundos de baja frecuencia.",
        "miembros": ["Ballena Azul", "Ballena de Aleta", "Ballena Enana", "Ballena Sei"],
        "caracteristicas": "Barbas para filtrar alimento, cantos profundos, migración anual"
    },
    "Megapteridae": {
        "nombre": "Ballena Jorobada",
        "descripcion": "Familia con una sola especie. Conocida por sus complejos cantos y comportamiento acrobático. Muy social.",
        "miembros": ["Ballena Jorobada"],
        "caracteristicas": "Cantos complejos, acrobacias acuáticas, aletas largas"
    },
    "Balaenidae": {
        "nombre": "Ballenas Francas",
        "descripcion": "Ballenas barbadas lentas. Muy amenazadas. Emiten sonidos más complejos que otros rorcuales.",
        "miembros": ["Ballena Franca del Atlántico Norte"],
        "caracteristicas": "Sonidos complejos, estructura social fuerte, migración"
    },
    "Ziphiidae": {
        "nombre": "Ballenas Pico (Beaked Whales)",
        "descripcion": "Cetáceos profundos con capacidad de ecolocalización. Menos estudiadas que otras familias.",
        "miembros": ["Ballena Pico del Norte"],
        "caracteristicas": "Ecolocalización, inmersión profunda, clics de alta frecuencia"
    },
    "Delphinidae": {
        "nombre": "Delfines y Orcas",
        "descripcion": "Cetáceos dentados altamente inteligentes y sociales. Excelentes comunicadores con grandes repertorios de sonidos.",
        "miembros": ["Orca", "Ballena Piloto"],
        "caracteristicas": "Clics y silbidos, alta inteligencia, comportamiento social complejo"
    },
    "Physeteridae": {
        "nombre": "Cachalotes",
        "descripcion": "Cetáceos dentados buceadores profundos. Cerebro más grande del reino animal. Comunicación mediante codas rítmicas.",
        "miembros": ["Cachalote"],
        "caracteristicas": "Clics rítmicos, buceo profundo, inteligencia superior"
    }
}

# Función para obtener la familia de una especie
def get_cetacean_family(prediction):
    """Retorna la información de la familia de una especie"""
    if prediction in CETACEAN_INFO:
        family_name = CETACEAN_INFO[prediction].get("familia", "Desconocida")
        return family_name, FAMILY_INFO.get(family_name, {})
    return None, {}

# Función para extraer rango numérico de una cadena de frecuencia
def extract_frequency_range(freq_string):
    """Extrae el rango mínimo y máximo de una cadena de frecuencia
    Ejemplo: 'Baja (50-1000 Hz)' -> (50, 1000)"""
    import re
    match = re.search(r'(\d+)\s*-\s*(\d+)\s*Hz', freq_string)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

# Función para determinar si dos rangos de frecuencia se solapan
def frequency_ranges_overlap(range1, range2):
    """Determina si dos rangos de frecuencia tienen solapamiento
    que permita comunicación acústica"""
    min1, max1 = extract_frequency_range(range1)
    min2, max2 = extract_frequency_range(range2)

    if min1 is None or min2 is None:
        return False

    # Hay solapamiento si el máximo de uno es mayor o igual al mínimo del otro
    # y viceversa
    overlap = not (max1 < min2 or max2 < min1)
    return overlap

# Función para determinar solapamiento con datos numéricos reales
def frequency_ranges_overlap_real(min1, max1, min2, max2):
    """Determina si dos rangos de frecuencia numéricos se solapan"""
    if min1 is None or max1 is None or min2 is None or max2 is None:
        return False

    # Hay solapamiento si el máximo de uno es mayor o igual al mínimo del otro
    overlap = not (max1 < min2 or max2 < min1)
    return overlap

# Función para evaluar la comunicación entre dos especies
def analyze_communication(prediction1, prediction2):
    """Analiza si dos especies pueden comunicarse entre sí basándose en:
    1. Si pertenecen a la misma familia
    2. Si sus rangos de frecuencia se solapan
    3. Si sus características acústicas son compatibles"""
    if prediction1 not in CETACEAN_INFO or prediction2 not in CETACEAN_INFO:
        return None

    info1 = CETACEAN_INFO[prediction1]
    info2 = CETACEAN_INFO[prediction2]

    family1 = info1.get("familia")
    family2 = info2.get("familia")

    same_family = family1 == family2

    # Usar datos de frecuencia reales si están disponibles
    if 'rango_frecuencia_real' in info1 and 'rango_frecuencia_real' in info2:
        # Extraer rangos numéricos reales
        min1 = info1.get('frecuencia_media_real', 0) - 100  # Margen de tolerancia
        max1 = info1.get('frecuencia_media_real', 0) + 100
        min2 = info2.get('frecuencia_media_real', 0) - 100
        max2 = info2.get('frecuencia_media_real', 0) + 100

        # Usar rangos reales si están disponibles
        real_min1 = info1.get('rango_min_hz', min1)
        real_max1 = info1.get('rango_max_hz', max1)
        real_min2 = info2.get('rango_min_hz', min2)
        real_max2 = info2.get('rango_max_hz', max2)

        can_hear = frequency_ranges_overlap_real(real_min1, real_max1, real_min2, real_max2)
        range1 = f"{real_min1:.1f}-{real_max1:.1f} Hz (real)"
        range2 = f"{real_min2:.1f}-{real_max2:.1f} Hz (real)"
    else:
        # Usar datos hardcodeados como fallback
        range1 = info1.get("rango_frecuencia", "")
        range2 = info2.get("rango_frecuencia", "")
        can_hear = frequency_ranges_overlap(range1, range2)

    # Si pertenecen a la misma familia, es más probable que puedan comunicarse
    # incluso si hay poca diferencia en frecuencias
    if not can_hear and same_family:
        can_hear = True

    # Calcular porcentaje de solapamiento si ambos tienen datos reales
    overlap_percentage = 0
    if 'rango_frecuencia_real' in info1 and 'rango_frecuencia_real' in info2:
        real_min1 = info1.get('rango_min_hz', 0)
        real_max1 = info1.get('rango_max_hz', 0)
        real_min2 = info2.get('rango_min_hz', 0)
        real_max2 = info2.get('rango_max_hz', 0)

        # Calcular solapamiento
        overlap_start = max(real_min1, real_min2)
        overlap_end = min(real_max1, real_max2)

        if overlap_end > overlap_start:
            overlap_range = overlap_end - overlap_start
            total_range = max(real_max1, real_max2) - min(real_min1, real_min2)
            if total_range > 0:
                overlap_percentage = (overlap_range / total_range) * 100

    return {
        "same_family": same_family,
        "family1": family1,
        "family2": family2,
        "range1": range1,
        "range2": range2,
        "can_hear": can_hear,
        "overlap": frequency_ranges_overlap(range1, range2) if 'rango_frecuencia_real' not in info1 else frequency_ranges_overlap_real(real_min1, real_max1, real_min2, real_max2),
        "overlap_percentage": overlap_percentage,
        "using_real_data": 'rango_frecuencia_real' in info1 and 'rango_frecuencia_real' in info2
    }


# Función para extraer características del audio
def extract_features(file_path):
    """Extrae características MFCC del archivo de audio"""
    try:
        y, sr = librosa.load(file_path)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features = np.mean(mfccs.T, axis=0)
        return features
    except Exception as e:
        st.error(f"Error al procesar el audio: {e}")
        return None

# Función para extraer el espectrograma y frecuencias dominantes
def extract_frequency_spectrum(file_path):
    """Extrae el espectrograma y las frecuencias dominantes del audio"""
    try:
        y, sr = librosa.load(file_path)
        
        # Calcular magnitud del espectrograma
        D = librosa.stft(y)
        S = np.abs(D)
        
        # Convertir a escala de frecuencia
        freqs = librosa.fft_frequencies(sr=sr)
        
        # Energía promedio por frecuencia
        magnitude = np.mean(S, axis=1)
        
        # Encontrar picos de frecuencia
        threshold = np.max(magnitude) * 0.1  # 10% del máximo
        peaks = np.where(magnitude > threshold)[0]
        dominant_freqs = freqs[peaks]
        
        # Estadísticas
        min_freq = np.min(dominant_freqs) if len(dominant_freqs) > 0 else 0
        max_freq = np.max(dominant_freqs) if len(dominant_freqs) > 0 else 0
        mean_freq = np.mean(dominant_freqs) if len(dominant_freqs) > 0 else 0
        
        return {
            'sr': sr,
            'freqs': freqs,
            'magnitude': magnitude,
            'S': S,
            'min_freq': min_freq,
            'max_freq': max_freq,
            'mean_freq': mean_freq,
            'dominant_freqs': dominant_freqs
        }
    except Exception as e:
        st.error(f"Error al analizar espectrograma: {e}")
        return None

# Función para crear visualización del espectrograma
def plot_spectrogram(file_path, title="Espectrograma"):
    """Crea una visualización del espectrograma del audio"""
    try:
        y, sr = librosa.load(file_path)
        
        fig, ax = plt.subplots(figsize=(12, 4))
        
        # Crear espectrograma en escala Mel
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_db = librosa.power_to_db(S, ref=np.max)
        
        # Mostrar
        img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', ax=ax)
        ax.set_title(title, fontsize=12, fontweight='bold')
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error al crear espectrograma: {e}")
        return None

# Cargar el modelo
@st.cache_resource
def load_model():
    """Carga el modelo de clasificación"""
    model_path = "/home/azureuser/cloudfiles/code/Users/alvaro.lopezredondo.3431/clasificador_ballenas.pkl"
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"❌ Modelo no encontrado en {model_path}")
        return None
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {e}")
        return None

# Función para obtener información del modelo
def get_model_info(model):
    """Extrae información detallada del modelo ML"""
    info = {}
    try:
        info['tipo'] = type(model).__name__
        info['n_features'] = model.n_features_in_
        info['n_clases'] = len(model.classes_)
        info['clases'] = list(model.classes_)
        info['n_estimadores'] = model.n_estimators if hasattr(model, 'n_estimators') else 'N/A'
        info['max_profundidad'] = model.max_depth if hasattr(model, 'max_depth') else 'N/A'
        info['importancia_features'] = model.feature_importances_ if hasattr(model, 'feature_importances_') else None
    except Exception as e:
        st.error(f"Error extrayendo información del modelo: {e}")
    return info

# Interfaz principal
st.markdown("""
<h1 style='color:#1976d2; font-size:2.5rem; font-weight:700; margin-bottom:0.5em;'>Clasificador de Cetáceos por Sonido</h1>
<p style='font-size:1.2rem;'>Este clasificador utiliza inteligencia artificial para identificar especies de cetáceos basándose en sus sonidos únicos. Carga un archivo de audio de una ballena o delfín para obtener la predicción.</p>
""", unsafe_allow_html=True)

# Cargar el modelo
model = load_model()

if model is None:
    st.error("❌ No se puede cargar el modelo. Por favor, verifica que el archivo existe.")
    st.stop()

# Botón en la barra lateral para mostrar información del modelo
with st.sidebar:
    st.markdown("<h2 style='color:#1976d2; font-size:1.3rem;'>Opciones</h2>", unsafe_allow_html=True)
    st.divider()
    if st.button("Información del Modelo ML", use_container_width=True, key="btn_model_info"):
        st.session_state.show_model_info = True
        st.session_state.show_advanced_analysis = False
    if st.button("📊 Análisis Avanzado", use_container_width=True, key="btn_advanced_analysis"):
        st.session_state.show_advanced_analysis = True
        st.session_state.show_model_info = False
    if st.button("Volver a Inicio", use_container_width=True, key="btn_home"):
        st.session_state.show_model_info = False
        st.session_state.show_advanced_analysis = False
        st.rerun()

if st.session_state.get('show_model_info', False):
    st.markdown("<h2 style='color:#1976d2;'>Información Detallada del Modelo Machine Learning</h2>", unsafe_allow_html=True)
    
    try:
        model_info = get_model_info(model)
        
        # Crear tres columnas para mostrar información
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Tipo de Modelo", model_info['tipo'])
            st.metric("Características (Features)", model_info['n_features'])
        
        with col2:
            st.metric("Clases Soportadas", model_info['n_clases'])
            st.metric("Estimadores (Árboles)", model_info['n_estimadores'])
        
        with col3:
            st.metric("Profundidad Máxima", model_info['max_profundidad'])
            st.metric("Estado", "✅ Listo")
        
        st.divider()
        
        # Información detallada del modelo
        st.markdown("<h3 style='color:#424242;'>Características del Modelo</h3>", unsafe_allow_html=True)
        st.write(f"""
        - **Algoritmo**: Random Forest (Ensemble de Árboles de Decisión)
        - **Número de Features (MFCC)**: {model_info['n_features']}
        - **Número de Clases**: {model_info['n_clases']}
        - **Número de Estimadores**: {model_info['n_estimadores']}
        - **Profundidad Máxima**: {model_info['max_profundidad']}
        """)
        
        st.markdown("<h3 style='color:#424242;'>Especies Clasificadas</h3>", unsafe_allow_html=True)
        
        # Crear dos columnas para listar especies
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Primera mitad:**")
            for i, clase in enumerate(model_info['clases'][:len(model_info['clases'])//2 + 1], 1):
                nombre_comun = CETACEAN_INFO.get(clase, {}).get("nombre_comun", clase)
                color = CETACEAN_INFO.get(clase, {}).get("color", "#1976d2")
                st.markdown(f"<div style='display:flex;align-items:center;'><div style='width:16px;height:16px;background:{color};border-radius:50%;margin-right:8px;'></div> <span>{i}. {nombre_comun}</span></div>", unsafe_allow_html=True)
        with col2:
            st.write("**Segunda mitad:**")
            for i, clase in enumerate(model_info['clases'][len(model_info['clases'])//2 + 1:], len(model_info['clases'])//2 + 2):
                nombre_comun = CETACEAN_INFO.get(clase, {}).get("nombre_comun", clase)
                color = CETACEAN_INFO.get(clase, {}).get("color", "#1976d2")
                st.markdown(f"<div style='display:flex;align-items:center;'><div style='width:16px;height:16px;background:{color};border-radius:50%;margin-right:8px;'></div> <span>{i}. {nombre_comun}</span></div>", unsafe_allow_html=True)
        
        st.divider()
        
        # Importancia de características
        if model_info['importancia_features'] is not None:
            st.markdown("<h3 style='color:#424242;'>Importancia de Características (MFCC)</h3>", unsafe_allow_html=True)
            
            importancia_df = pd.DataFrame({
                'Característica': [f'MFCC {i}' for i in range(model_info['n_features'])],
                'Importancia': model_info['importancia_features']
            }).sort_values('Importancia', ascending=False)
            
            # Gráfico de importancia
            st.bar_chart(importancia_df.set_index('Característica')['Importancia'])
            
            st.dataframe(importancia_df, use_container_width=True)
        
        st.divider()
        
        # Información sobre las características MFCC
        st.markdown("<h3 style='color:#424242;'>¿Qué son las Características MFCC?</h3>", unsafe_allow_html=True)
        st.info("""
        **MFCC (Mel-Frequency Cepstral Coefficients)** son características acústicas que representan:
        - **Escala Mel**: Mimética del oído humano (mejor percepción de frecuencias bajas)
        - **13 Coeficientes**: Capturan la "huella dactilar" del sonido
        - **Uso**: Bioacústica, reconocimiento de voz, clasificación de especies
        El modelo extrae estas 13 características de cada archivo de audio y las usa para identificar automáticamente la especie de cetáceo.
        """)
        
    except Exception as e:
        st.error(f"❌ Error al mostrar información del modelo: {e}")
    
    # Botón para volver
    st.divider()
    if st.button("← Volver a Clasificar", use_container_width=True):
        st.session_state.show_model_info = False
        st.rerun()

# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN: ANÁLISIS AVANZADO
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.get('show_advanced_analysis', False):
    st.markdown("""
    <h2 style='color:#1976d2;'>📊 Análisis Avanzado de Características Acústicas</h2>
    """, unsafe_allow_html=True)
    
    st.info("""
    Este análisis examina en profundidad las características acústicas (MFCCs - Mel-Frequency Cepstral Coefficients) 
    de 11 especies de cetáceos. Utiliza técnicas de machine learning para entender cómo varían sus vocalización entre especies y familias taxonómicas.
    """)
    
    # Base de datos de archivos de análisis
    analysis_files = [
        {
            "file": "analisis_correlacion_mfcc.png",
            "title": "1. Correlación entre Coeficientes MFCC",
            "description": "Muestra la matriz de correlación entre los 13 coeficientes MFCC. Las correlaciones altas entre MFCCs consecutivos indican redundancia espectral, lo que significa que capturan información similar sobre el sonido."
        },
        {
            "file": "analisis_distancias_acusticas.png",
            "title": "2. Matriz de Distancias Acústicas",
            "description": "Matriz que representa la similitud/diferencia acústica entre todas las especies de cetáceos. Las distancias más pequeñas indican species más similares acústicamente. Por ejemplo, las Ballenas Francas del Atlántico tienen distancias muy pequeñas entre sí."
        },
        {
            "file": "analisis_dendrograma.png",
            "title": "3. Dendrograma de Clustering Jerárquico",
            "description": "Árbol de agrupamiento que muestra cómo se agrupan las especies por similitud acústica sin usar información de familia taxonómica. Si los clusters coinciden con las familias reales, confirma que la acústica refleja la evolución de estas especies."
        },
        {
            "file": "analisis_pca_vs_umap.png",
            "title": "4. PCA vs UMAP: Reducciones Dimensionales",
            "description": "Comparación de dos técnicas de reducción dimensional: PCA (lineal) y UMAP (no-lineal). PCA es más estable con datasets pequeños. UMAP preserva mejor la topología local. Observa cómo ambos métodos agrupan similares especies juntas."
        },
        {
            "file": "analisis_feature_importance.png",
            "title": "5. Importancia de Features (MFCCs)",
            "description": "Ranking de qué coeficientes MFCC son más importantes para discriminar entre familias taxonómicas. MFCC_0 captura la energía global, mientras que los primeros coeficientes codifican la forma espectral gruesa (frecuencias dominantes)."
        },
        {
            "file": "analisis_confusion_matrix.png",
            "title": "6. Matriz de Confusión KNN (Leave-One-Out)",
            "description": "Muestra cuán bien un clasificador KNN puede distinguir entre familias taxonómicas usando solo características acústicas. La diagonal principal indica clasificaciones correctas. Valores altos fuera de la diagonal indican confusiones entre familias."
        },
        {
            "file": "analisis_radar_familias.png",
            "title": "7. Perfil Acústico Radar por Familia",
            "description": "Gráfico radar que muestra la 'firma acústica' de cada familia taxonómica usando los primeros 8 MFCCs. Formas muy distintas entre familias indican que están acústicamente bien diferenciadas."
        }
    ]
    
    # Path base donde están los gráficos
    base_path = "/home/azureuser/cloudfiles/code"
    
    # Mostrar cada gráfico con su descripción
    for analysis in analysis_files:
        file_path = os.path.join(base_path, analysis["file"])
        
        if os.path.exists(file_path):
            st.markdown(f"<h3 style='color:#1976d2;'>{analysis['title']}</h3>", unsafe_allow_html=True)
            st.markdown(analysis['description'])
            
            # Mostrar la imagen
            image = st.image(file_path, use_container_width=True)
            st.divider()
        else:
            st.warning(f"⚠️ Archivo no encontrado: {analysis['file']}")
    
    # Resumen de insights
    st.markdown("<h3 style='color:#1976d2;'>📌 Insights Principales</h3>", unsafe_allow_html=True)
    st.markdown("""
    - **Diferenciación Familiar**: Los MFCCs capturan diferencias acústicas reales entre familias de cetáceos
    - **MFCC_0 Crítico**: La energía del sonido (MFCC_0) es el discriminador más importante entre especies
    - **Agrupamiento Natural**: El clustering jerárquico muestra que las especies se agrupan naturalmente por familia, 
      sin necesidad de etiquetas previas
    - **Similitudes Esperadas**: Ballenas Francas (Balaenidae) son muy similares entre sí, así como los Delphinidae
    - **Acústica vs Taxonomía**: La similitud acústica correlaciona bien con la clasificación taxonómica,
      sugiriendo que la evolución ha moldeado las vocalizaciones de manera familiar
    """)
    
    st.divider()
    if st.button("← Volver a Clasificar", use_container_width=True):
        st.session_state.show_advanced_analysis = False
        st.rerun()

# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN: INTERFAZ PRINCIPAL (Clasificador)
# ═════════════════════════════════════════════════════════════════════════════
else:
    st.markdown("<h3 style='color:#1976d2;'>Carga tus Audios</h3>", unsafe_allow_html=True)
    st.write("Puedes subir 1 o 2 audios de cetáceos. Si subes 2, podrás compararlos.")
    
    uploaded_files = st.file_uploader(
        "Selecciona archivos de audio (.wav, .mp3, .flac, etc.)",
        type=["wav", "mp3", "flac", "ogg"],
        key="audio_uploader",
        accept_multiple_files=True
    )

    if uploaded_files and len(uploaded_files) > 0:
        temp_dir = "/tmp"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        # Procesar audios
        results = []
        temp_paths = []
        
        try:
            with st.spinner("Analizando los audios... Por favor espera."):
                for idx, uploaded_file in enumerate(uploaded_files[:2]):  # Limitar a 2 archivos
                    temp_audio_path = os.path.join(temp_dir, f"temp_{idx}_{uploaded_file.name}")
                    temp_paths.append(temp_audio_path)
                    
                    # Guardar archivo
                    with open(temp_audio_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Extraer características
                    features = extract_features(temp_audio_path)
                    
                    if features is not None:
                        # Realizar predicción
                        features_reshaped = features.reshape(1, -1)
                        prediction = model.predict(features_reshaped)[0]
                        prediction_proba = model.predict_proba(features_reshaped)[0]
                        confidence = np.max(prediction_proba) * 100
                        
                        # Extraer espectrograma y frecuencias
                        spectrum = extract_frequency_spectrum(temp_audio_path)
                        
                        results.append({
                            'file': uploaded_file,
                            'prediction': prediction,
                            'confidence': confidence,
                            'probabilities': prediction_proba,
                            'features': features,
                            'spectrum': spectrum
                        })
            
            # Mostrar resultados
            if len(results) == 1:
                # Caso: Un solo audio
                result = results[0]
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("<h3 style='color:#1976d2;'>Audio Cargado</h3>", unsafe_allow_html=True)
                    st.audio(result['file'], format=f"audio/{result['file'].name.split('.')[-1]}")
                
                with col2:
                    st.markdown("<h3 style='color:#1976d2;'>Resultado de la Predicción</h3>", unsafe_allow_html=True)
                    if result['prediction'] in CETACEAN_INFO:
                        info = CETACEAN_INFO[result['prediction']]
                        st.markdown(f"<div style='padding:1em;border-radius:8px;background:#e3f2fd;margin-bottom:1em;'><span style='font-size:1.3rem;font-weight:600;color:#1976d2;'>{info['nombre_comun']}</span><br><span style='color:#616161;'>Confianza: {result['confidence']:.1f}%</span></div>", unsafe_allow_html=True)
                        st.markdown(f"**Nombre científico:** *{info['nombre_cientifico']}*", unsafe_allow_html=True)
                        st.markdown(f"**Descripción:**\n\n{info['descripcion']}", unsafe_allow_html=True)

                        # Mostrar información de frecuencia
                        st.markdown("**📡 Información Acústica:**")
                        if 'rango_frecuencia_real' in info:
                            st.markdown(f"- **Rango de frecuencia real:** {info['rango_frecuencia_real']}")
                            st.markdown(f"- **Frecuencia media:** {info['frecuencia_media_real']:.1f} Hz")
                            st.markdown(f"- **Frecuencia pico:** {info['frecuencia_pico_real']:.1f} Hz")
                        else:
                            st.markdown(f"- **Rango de frecuencia estimado:** {info['rango_frecuencia']}")
                        st.markdown(f"- **Tipo de voz:** {info['tipo_voz']}")
                        st.markdown(f"- **Familia:** {info['familia']}")

                        # Mostrar espectrograma si está disponible
                        if result['spectrum']:
                            st.markdown("**🎵 Espectrograma del Audio:**")
                            fig = plot_spectrogram(temp_audio_path, f"Espectrograma - {info['nombre_comun']}")
                            if fig:
                                st.pyplot(fig)
                                plt.close(fig)
                
                # Mostrar probabilidades
                st.divider()
                st.markdown("<h3 style='color:#424242;'>Probabilidades de Todas las Especies</h3>", unsafe_allow_html=True)
                
                classes = model.classes_
                probabilities = result['probabilities'] * 100
                
                prob_df = pd.DataFrame({
                    "Especie": [CETACEAN_INFO.get(c, {}).get("nombre_comun", c) for c in classes],
                    "Probabilidad (%)": probabilities
                }).sort_values("Probabilidad (%)", ascending=False)
                
                st.bar_chart(prob_df.set_index("Especie")["Probabilidad (%)"])
                st.dataframe(prob_df, use_container_width=True)
            
            elif len(results) == 2:
                # Caso: Dos audios - Mostrar comparación
                st.markdown("<h2 style='color:#1976d2;'>📊 Comparación de Dos Cetáceos</h2>", unsafe_allow_html=True)
                
                # Mostrar audios en dos columnas
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("<h4>Audio 1</h4>", unsafe_allow_html=True)
                    st.audio(results[0]['file'], format=f"audio/{results[0]['file'].name.split('.')[-1]}")
                with col2:
                    st.markdown("<h4>Audio 2</h4>", unsafe_allow_html=True)
                    st.audio(results[1]['file'], format=f"audio/{results[1]['file'].name.split('.')[-1]}")
                
                st.divider()
                
                # Predicciones de ambos
                st.markdown("<h3 style='color:#1976d2;'>Resultados de las Predicciones</h3>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    result = results[0]
                    if result['prediction'] in CETACEAN_INFO:
                        info = CETACEAN_INFO[result['prediction']]
                        st.markdown(f"<div style='padding:1em;border-radius:8px;background:#e3f2fd;'><span style='font-size:1.2rem;font-weight:600;color:#1976d2;'>{info['nombre_comun']}</span><br><span style='color:#616161;'>Confianza: {result['confidence']:.1f}%</span></div>", unsafe_allow_html=True)
                        st.markdown(f"**Nombre científico:** *{info['nombre_cientifico']}*")
                        # Mostrar información de frecuencia
                        if 'rango_frecuencia_real' in info:
                            st.markdown(f"**Rango frecuencia real:** {info['rango_frecuencia_real']}")
                        else:
                            st.markdown(f"**Rango frecuencia:** {info['rango_frecuencia']}")
                        st.markdown(f"**Familia:** {info['familia']}")
                
                with col2:
                    result = results[1]
                    if result['prediction'] in CETACEAN_INFO:
                        info = CETACEAN_INFO[result['prediction']]
                        st.markdown(f"<div style='padding:1em;border-radius:8px;background:#f3e5f5;'><span style='font-size:1.2rem;font-weight:600;color:#7b1fa2;'>{info['nombre_comun']}</span><br><span style='color:#616161;'>Confianza: {result['confidence']:.1f}%</span></div>", unsafe_allow_html=True)
                        st.markdown(f"**Nombre científico:** *{info['nombre_cientifico']}*")
                        # Mostrar información de frecuencia
                        if 'rango_frecuencia_real' in info:
                            st.markdown(f"**Rango frecuencia real:** {info['rango_frecuencia_real']}")
                        else:
                            st.markdown(f"**Rango frecuencia:** {info['rango_frecuencia']}")
                        st.markdown(f"**Familia:** {info['familia']}")
                
                st.divider()
                
                # Comparación detallada
                st.markdown("<h3 style='color:#424242;'>Análisis Comparativo</h3>", unsafe_allow_html=True)
                
                info1 = CETACEAN_INFO.get(results[0]['prediction'], {})
                info2 = CETACEAN_INFO.get(results[1]['prediction'], {})
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**{info1.get('nombre_comun', 'Desconocido')}**")
                    st.write(f"📋 {info1.get('descripcion', 'N/A')[:600]}")
                    st.metric("Confianza en la Predicción", f"{results[0]['confidence']:.1f}%")
                
                with col2:
                    st.markdown(f"**{info2.get('nombre_comun', 'Desconocido')}**")
                    st.write(f"📋 {info2.get('descripcion', 'N/A')[:600]}")
                    st.metric("Confianza en la Predicción", f"{results[1]['confidence']:.1f}%")
                
                st.divider()
                
                # Análisis de Familias y Comunicación
                st.markdown("<h3 style='color:#1976d2;'>🔬 Relación Entre Familias y Capacidad de Comunicación</h3>", unsafe_allow_html=True)
                
                family1 = info1.get('familia', 'Desconocida')
                family2 = info2.get('familia', 'Desconocida')
                
                comm_analysis = analyze_communication(results[0]['prediction'], results[1]['prediction'])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**{info1.get('nombre_comun', 'Especie 1')}**")
                    st.markdown(f"- 👨‍👩‍👧‍👦 Familia: **{family1}**")
                    if family1 in FAMILY_INFO:
                        family_data = FAMILY_INFO[family1]
                        st.markdown(f"- 📝 {family_data.get('caracteristicas', 'N/A')}")
                
                with col2:
                    st.markdown(f"**{info2.get('nombre_comun', 'Especie 2')}**")
                    st.markdown(f"- 👨‍👩‍👧‍👦 Familia: **{family2}**")
                    if family2 in FAMILY_INFO:
                        family_data = FAMILY_INFO[family2]
                        st.markdown(f"- 📝 {family_data.get('caracteristicas', 'N/A')}")
                
                st.divider()
                
                # Información sobre relación familiar y comunicación
                if comm_analysis:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if comm_analysis['same_family']:
                            st.success("✅ **Misma Familia**\nComparten origen evolutivo")
                        else:
                            st.info("❌ **Familias Diferentes**\nOrigen evolutivo distinto")
                    
                    with col2:
                        # Mostrar rangos de frecuencia usando datos reales
                        if comm_analysis['using_real_data']:
                            st.markdown("**📡 Rangos de Frecuencia (Datos Reales):**")
                            st.markdown(f"**{info1.get('nombre_comun')}:** {comm_analysis['range1']}")
                            st.markdown(f"**{info2.get('nombre_comun')}:** {comm_analysis['range2']}")
                            if comm_analysis.get('overlap_percentage', 0) > 0:
                                st.metric("Solapamiento", f"{comm_analysis['overlap_percentage']:.1f}%")
                        else:
                            rango1 = info1.get('rango_frecuencia', 'Desconocido')
                            rango2 = info2.get('rango_frecuencia', 'Desconocido')
                            st.markdown(f"**Rango Frecuencia 1:** {rango1}\n\n**Rango Frecuencia 2:** {rango2}")
                    
                    with col3:
                        if comm_analysis['can_hear']:
                            if comm_analysis['using_real_data']:
                                st.success("✅ **Pueden Comunicarse**\nAnálisis basado en espectrograma real")
                            else:
                                st.success("✅ **Pueden Comunicarse**\nRangos de frecuencia compatibles")
                        else:
                            if comm_analysis['using_real_data']:
                                st.warning("⚠️ **Difícil Comunicación**\nFrecuencias muy diferentes (espectrograma)")
                            else:
                                st.warning("⚠️ **Difícil Comunicación**\nRangos de frecuencia muy diferentes")
                
                # Información adicional sobre comunicación
                if comm_analysis and comm_analysis['using_real_data']:
                    st.markdown("### 🔊 **Análisis Acústico Detallado**")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**{info1.get('nombre_comun')}**")
                        if 'frecuencia_media_real' in info1:
                            st.metric("Frecuencia Media", f"{info1['frecuencia_media_real']:.1f} Hz")
                        if 'frecuencia_pico_real' in info1:
                            st.metric("Frecuencia Pico", f"{info1['frecuencia_pico_real']:.1f} Hz")
                    
                    with col2:
                        st.markdown(f"**{info2.get('nombre_comun')}**")
                        if 'frecuencia_media_real' in info2:
                            st.metric("Frecuencia Media", f"{info2['frecuencia_media_real']:.1f} Hz")
                        if 'frecuencia_pico_real' in info2:
                            st.metric("Frecuencia Pico", f"{info2['frecuencia_pico_real']:.1f} Hz")
                    
                    # Explicación del análisis
                    if comm_analysis['can_hear']:
                        st.success("💬 **Compatibilidad Acústica:** Estas especies pueden escucharse mutuamente. El análisis del espectrograma muestra que sus frecuencias vocales se solapan lo suficiente para permitir comunicación.")
                    else:
                        st.warning("🚫 **Barrera Acústica:** Estas especies tienen dificultades para comunicarse. Sus frecuencias vocales son muy diferentes según el análisis del espectrograma.")
                    
                    if comm_analysis['same_family']:
                        st.info("👨‍👩‍👧‍👦 **Ventaja Familiar:** Al pertenecer a la misma familia, es posible que compartan patrones de vocalización similares a pesar de las diferencias en frecuencia.")
                
                st.divider()
                
                # Visualizar espectrogramas
                st.markdown("<h3 style='color:#1976d2;'>🎵 Espectrogramas de los Audios</h3>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                # Espectrograma 1
                with col1:
                    st.markdown(f"**{info1.get('nombre_comun')}**")
                    fig1 = plot_spectrogram(temp_paths[0], f"Espectrograma - {info1.get('nombre_comun')}")
                    if fig1:
                        st.pyplot(fig1)
                        plt.close(fig1)
                
                # Espectrograma 2
                with col2:
                    st.markdown(f"**{info2.get('nombre_comun')}**")
                    fig2 = plot_spectrogram(temp_paths[1], f"Espectrograma - {info2.get('nombre_comun')}")
                    if fig2:
                        st.pyplot(fig2)
                        plt.close(fig2)
                
                st.divider()
                
                # Análisis detallado de solapamiento de frecuencias
                st.markdown("<h4 style='color:#424242;'>📡 Análisis Detallado de Solapamiento de Frecuencias</h4>", unsafe_allow_html=True)
                
                if comm_analysis:
                    # Extraer rangos numéricos esperados
                    min1_exp, max1_exp = extract_frequency_range(comm_analysis['range1'])
                    min2_exp, max2_exp = extract_frequency_range(comm_analysis['range2'])
                    
                    # Extraer rangos reales del espectrograma
                    if results[0]['spectrum'] and results[1]['spectrum']:
                        spec1 = results[0]['spectrum']
                        spec2 = results[1]['spectrum']
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"**{info1.get('nombre_comun')}**")
                            st.markdown("*Rango Esperado (Literatura):*")
                            if min1_exp and max1_exp:
                                st.info(f"📊 **{min1_exp} - {max1_exp} Hz**")
                            st.markdown("*Rango Real Detectado (Espectrograma):*")
                            st.success(f"📈 **{spec1['min_freq']:.1f} - {spec1['max_freq']:.1f} Hz**")
                            st.caption(f"Media: {spec1['mean_freq']:.1f} Hz")
                        
                        with col2:
                            st.markdown(f"**{info2.get('nombre_comun')}**")
                            st.markdown("*Rango Esperado (Literatura):*")
                            if min2_exp and max2_exp:
                                st.info(f"📊 **{min2_exp} - {max2_exp} Hz**")
                            st.markdown("*Rango Real Detectado (Espectrograma):*")
                            st.success(f"📈 **{spec2['min_freq']:.1f} - {spec2['max_freq']:.1f} Hz**")
                            st.caption(f"Media: {spec2['mean_freq']:.1f} Hz")
                        
                        st.divider()
                        
                        # Analizar solapamiento REAL
                        real_overlap_min = max(spec1['min_freq'], spec2['min_freq'])
                        real_overlap_max = min(spec1['max_freq'], spec2['max_freq'])
                        
                        st.markdown("**Comparación de Solapamientos:**")
                        
                        if real_overlap_max >= real_overlap_min:
                            st.success(f"✅ **Solapamiento REAL:** {real_overlap_min:.1f} - {real_overlap_max:.1f} Hz")
                            st.markdown(f"""
                            Los espectrogramas demuestran que **PUEDEN COMUNICARSE** en el rango real de **{real_overlap_min:.1f} - {real_overlap_max:.1f} Hz**.
                            
                            **Coincidencia:** Ambas especies producen sonidos que el otra especie puede escuchar.
                            """)
                        else:
                            st.warning(f"⚠️ **Solapamiento Limitado**")
                    
                    # Visualizar solapamiento esperado
                    if min1_exp and max1_exp and min2_exp and max2_exp:
                        overlap_min = max(min1_exp, min2_exp)
                        overlap_max = min(max1_exp, max2_exp)
                        
                        if overlap_max >= overlap_min:
                            st.info(f"📋 Solapamiento ESPERADO (literatura): {overlap_min} - {overlap_max} Hz")
                        else:
                            st.warning(f"📋 Sin solapamiento esperado según rango de literatura")
                
                st.divider()
                
                # Detalles sobre familias
                st.markdown("<h4 style='color:#424242;'>📚 Información sobre las Familias</h4>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if family1 in FAMILY_INFO:
                        fam_info = FAMILY_INFO[family1]
                        st.markdown(f"**{fam_info['nombre']}**")
                        st.write(fam_info['descripcion'])
                        st.markdown(f"**Miembros en el clasificador:** {', '.join(fam_info['miembros'])}")
                
                with col2:
                    if family2 in FAMILY_INFO:
                        fam_info = FAMILY_INFO[family2]
                        st.markdown(f"**{fam_info['nombre']}**")
                        st.write(fam_info['descripcion'])
                        st.markdown(f"**Miembros en el clasificador:** {', '.join(fam_info['miembros'])}")
                
                st.divider()
                
                # Comparación de características acústicas
                st.markdown("<h3 style='color:#424242;'>Comparación de Características Acústicas (MFCC)</h3>", unsafe_allow_html=True)
                
                comparison_df = pd.DataFrame({
                    'Característica': [f'MFCC {i}' for i in range(len(results[0]['features']))],
                    info1.get('nombre_comun', 'Audio 1'): results[0]['features'],
                    info2.get('nombre_comun', 'Audio 2'): results[1]['features'],
                    'Diferencia': np.abs(results[0]['features'] - results[1]['features'])
                })
                
                st.dataframe(comparison_df, use_container_width=True)
                
                # Gráfico de comparación
                st.line_chart(comparison_df.set_index('Característica')[[info1.get('nombre_comun', 'Audio 1'), info2.get('nombre_comun', 'Audio 2')]])
                
                # Conclusión
                st.divider()
                st.markdown("<h3 style='color:#1976d2;'>Conclusión</h3>", unsafe_allow_html=True)
                
                if results[0]['prediction'] == results[1]['prediction']:
                    st.success(f"✅ Ambos audios corresponden a la **misma especie**: {info1.get('nombre_comun', 'Desconocido')}")
                else:
                    similarity = 1 - (np.mean(comparison_df['Diferencia']) / np.max(np.abs(results[0]['features'])))
                    st.info(f"🔄 Los audios corresponden a **diferentes especies**:\n\n- **Audio 1:** {info1.get('nombre_comun', 'Desconocido')}\n- **Audio 2:** {info2.get('nombre_comun', 'Desconocido')}\n\nLas características acústicas son bastante diferentes, lo que refleja las variaciones evolutivas entre estas especies.")
        
        except Exception as e:
            st.error(f"❌ Error al procesar los archivos: {str(e)}")
        
        finally:
            # Limpiar archivos temporales
            for temp_path in temp_paths:
                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except:
                    pass

    # Información adicional
    st.divider()
    st.markdown("<h3 style='color:#1976d2;'>Información sobre el Clasificador</h3>", unsafe_allow_html=True)
    
    with st.expander("¿Cómo funciona?"):
        st.write("""
        1. **Extracción de Características**: El sistema analiza el audio y extrae características acústicas (MFCC - Mel-Frequency Cepstral Coefficients)
        2. **Análisis**: Utiliza un modelo de Machine Learning (Random Forest) entrenado con sonidos de diferentes especies de cetáceos
        3. **Predicción**: Compara las características del audio con su base de datos para identificar la especie
        4. **Resultado**: Muestra la especie identificada, la confianza y detalles sobre el cetáceo
        """)
    
    with st.expander("Especies Soportadas"):
        for species, info in CETACEAN_INFO.items():
            color = info['color']
            st.markdown(f"<div style='display:flex;align-items:center;'><div style='width:16px;height:16px;background:{color};border-radius:50%;margin-right:8px;'></div> <b>{info['nombre_comun']}</b> - <i>{info['nombre_cientifico']}</i></div>", unsafe_allow_html=True)
    
    with st.expander("🔬 Familias de Cetáceos y Sus Relaciones"):
        st.markdown("""
        Los cetáceos se dividen en dos grandes grupos según su anatomía:
        
        ### 🐋 **Cetáceos Barbados (Misticetos)**
        Estos cetáceos tienen **barbas** en lugar de dientes para filtrar el agua.
        
        **Familias:**
        - **Balaenopteridae (Rorcuales)**: Ballenas grandes con acrobacias acuáticas
          - Incluyen: Ballena Azul, Ballena de Aleta, Ballena Enana, Ballena Sei
          - Cantos profundos de muy baja frecuencia (10-400 Hz)
          - **Relación:** Comparten origen evolutivo cercano, similar anatomía y comportamiento migratorio
        
        - **Megapteridae**: La Ballena Jorobada
          - Cantos complejos y melodías elaboradas
          - Aletas largas y comportamiento acrobático
          - **Relación:** Evolutivamente cercana a Balaenopteridae
        
        - **Balaenidae (Ballenas Francas)**: Ballenas lentas y robustas
          - Menos áil que rorcuales, emiten sonidos más complejos
          - **Relación:** Familia más antigua, menos relacionada con rorcuales modernos
        
        **¿Pueden comunicarse los cetáceos barbados entre sí?**
        - ✅ **Sí parcialmente**: Comparten rangos de frecuencia similares (baja frecuencia)
        - Las ballenas francas pueden escuchar a los rorcuales, pero los patrones de vocalización son diferentes
        - Normalmente **no interactúan socialmente** entre familias diferentes en la naturaleza
        
        ### 🐬 **Cetáceos Dentados (Odontocetos)**
        Estos cetáceos tienen **dientes** para capturar presas y usan ecolocalización.
        
        **Familias:**
        - **Delphinidae (Delfines y Orcas)**: Los más inteligentes y sociales
          - Incluyen: Orca, Ballena Piloto
          - Utilizan clics, silbidos y pulsaciones (300-130,000 Hz)
          - **Relación:** Gran inteligencia, comportamiento social complejo, cazadores cooperativos
        
        - **Ziphiidae (Ballenas Pico)**: Cetáceos profundos especializados
          - Ecolocalización de alta frecuencia
          - Buceadores extremos (hasta 2000+ metros)
          - **Relación:** Especializadas en aguas profundas
        
        - **Physeteridae (Cachalotes)**: Gigantes dentados
          - Cerebro más grande del reino animal
          - Clics rítmicos característicos llamados "codas"
          - **Relación:** Cazadores de calamares gigantes, comportamiento social inteligente
        
        **¿Pueden comunicarse los cetáceos dentados entre sí?**
        - ✅ **Sí con más facilidad**: Comparten rangos de frecuencia más amplios
        - Los Delphinidae (delfines) se comunican frecuentemente entre diferentes especies
        - Las orcas incluso pueden comunicarse con otras familias de cetáceos dentados
        - **Mayor compatibilidad acústica** que los barbados
        
        ### 📡 **Comunicación Entre Grupos**
        
        **¿Pueden barbados y dentados comunicarse?**
        - ❌ **Generalmente NO**: Sus rangos de frecuencia son completamente diferentes
        - **Barbados**: Muy baja frecuencia (10-400 Hz)
        - **Dentados**: Media-Alta frecuencia (300-130,000 Hz)
        - Es como intentar comunicarse a través de un teléfono roto - simplemente no pueden escucharse mutuamente en el rango de sus vocalizaciones
        
        ### 🧬 **Relaciones Evolutivas**
        
        **Árbol filogenético simplificado:**
        ```
        Cetáceos (antepasado común hace ~50 millones de años)
        ├── Mysticeti (Barbados)
        │   ├── Balaenopteridae
        │   ├── Megapteridae
        │   └── Balaenidae
        └── Odontoceti (Dentados)
            ├── Delphinidae
            ├── Ziphiidae
            └── Physeteridae
        ```
        
        ### 💡 **Datos Interesantes**
        - Las orcas (Delphinidae) a menudo son depredadores de otras ballenas
        - Las ballenas jorobadas son las únicas que pueden escuchar a otras ballenas barbadas y responder con patrones similares
        - Los cachalotes tienen "dialectos" de clics diferentes según el grupo familiar
        - Ballenas de familias cercanas (ej: Azul y Aleta) tienen características acústicas más similares que familias lejanas
        """)
    
    with st.expander("🎙️ ¿Cómo Afecta la Familia a la Clasificación?"):
        st.write("""
        El modelo de Machine Learning utiliza características acústicas (MFCCs) que reflejan naturalmente 
        las diferencias entre familias de cetáceos:
        
        1. **Ballenas Barbadas**: Producen cantos profundos y resonantes que resultan en MFCCs con valores 
           muy bajos (más energía en frecuencias bajas)
        
        2. **Cetáceos Dentados**: Producen clics y silbidos complejos que resultan en MFCCs con valores 
           más altos distribuidos en múltiples coeficientes
        
        3. **Identificación de Familia**: Al analizar dos audios comparables en el clasificador, 
           podemos ver si pertenecen a familias relacionadas observando la similitud de sus características acústicas
        
        4. **Predictor de Comunicación**: Si dos especies comparten rangos de frecuencia similares, 
           es más probable que se escuchen mutuamente en el océano
        """)
    
    # Footer
    st.divider()
    st.markdown("""
    <hr>
    <span style='color:#616161;'>Nota: Este clasificador funciona mejor con audios claros de ballenas y delfines. La precisión depende de la calidad del audio y de cuán clara sea la vocalización.</span>
    """, unsafe_allow_html=True)

# Importar pandas al final si no está ya importado
if 'pd' not in dir():
    import pandas as pd
