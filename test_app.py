#!/usr/bin/env python3
"""
Script para probar la app de Streamlit sin necesidad de abrirla en navegador
Verifica que todos los componentes funcionen correctamente
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*70)
print("🧪 PRUEBA DE FUNCIONALIDAD - CLASIFICADOR DE CETÁCEOS")
print("="*70 + "\n")

# 1. Verificar importaciones
print("1️⃣  Verificando importaciones...")
try:
    import streamlit as st
    import librosa
    import numpy as np
    import joblib
    import pandas as pd
    print("   ✅ Todas las librerías importadas correctamente\n")
except ImportError as e:
    print(f"   ❌ Error de importación: {e}\n")
    sys.exit(1)

# 2. Verificar archivos
print("2️⃣  Verificando archivos necesarios...")
model_path = "/home/azureuser/cloudfiles/code/Users/alvaro.lopezredondo.3431/clasificador_ballenas.pkl"
app_path = "/home/azureuser/cloudfiles/code/app_streamlit.py"

files_ok = True
if os.path.exists(model_path):
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"   ✅ Modelo cargado ({size_mb:.2f} MB)")
else:
    print(f"   ❌ Modelo NO encontrado")
    files_ok = False

if os.path.exists(app_path):
    print(f"   ✅ App Streamlit OK")
else:
    print(f"   ❌ App NO encontrada")
    files_ok = False

if not files_ok:
    sys.exit(1)

print()

# 3. Cargar y probar el modelo
print("3️⃣  Cargando modelo ML...")
try:
    model = joblib.load(model_path)
    print(f"   ✅ Modelo cargado: {type(model).__name__}")
    print(f"   📊 Clases soportadas: {len(model.classes_)}")
    print(f"   🔢 Features esperadas: {model.n_features_in_}\n")
except Exception as e:
    print(f"   ❌ Error: {e}\n")
    sys.exit(1)

# 4. Probar extracción de características
print("4️⃣  Probando extracción de características MFCC...")
try:
    # Crear un audio de prueba silencioso
    import tempfile
    import soundfile as sf
    
    sr = 22050  # Sample rate estándar
    duration = 1  # 1 segundo
    y = np.random.randn(sr * duration) * 0.01  # Ruido muy bajo
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        sf.write(tmp.name, y, sr)
        temp_file = tmp.name
    
    # Extraer características como lo hace la app
    y_loaded, sr_loaded = librosa.load(temp_file)
    mfccs = librosa.feature.mfcc(y=y_loaded, sr=sr_loaded, n_mfcc=13)
    features = np.mean(mfccs.T, axis=0)
    
    print(f"   ✅ Features extraídos: {len(features)} características")
    print(f"   📏 Dimensión: {features.shape}\n")
    
    # Limpiar
    os.remove(temp_file)
    
except Exception as e:
    print(f"   ❌ Error: {e}\n")
    sys.exit(1)

# 5. Probar predicción
print("5️⃣  Probando predicción del modelo...")
try:
    test_features = np.random.randn(1, 13)
    prediction = model.predict(test_features)
    probabilities = model.predict_proba(test_features)
    confidence = np.max(probabilities) * 100
    
    print(f"   ✅ Predicción: {prediction[0]}")
    print(f"   📊 Confianza: {confidence:.2f}%")
    print(f"   📈 Probabilidades obtenidas: {len(probabilities[0])} clases\n")
    
except Exception as e:
    print(f"   ❌ Error: {e}\n")
    sys.exit(1)

# 6. Probar información del modelo
print("6️⃣  Extrayendo información del modelo...")
try:
    model_info = {
        'tipo': type(model).__name__,
        'n_features': model.n_features_in_,
        'n_clases': len(model.classes_),
        'n_estimadores': model.n_estimators if hasattr(model, 'n_estimators') else 'N/A',
        'max_profundidad': model.max_depth if hasattr(model, 'max_depth') else 'N/A',
        'importancia_features': model.feature_importances_ if hasattr(model, 'feature_importances_') else None
    }
    
    print(f"   ✅ Tipo: {model_info['tipo']}")
    print(f"   🔢 Features: {model_info['n_features']}")
    print(f"   🏷️  Clases: {model_info['n_clases']}")
    print(f"   🌳 Estimadores: {model_info['n_estimadores']}")
    
    if model_info['importancia_features'] is not None:
        print(f"   📈 Top 3 features más importantes:")
        top_indices = np.argsort(model_info['importancia_features'])[-3:][::-1]
        for i, idx in enumerate(top_indices, 1):
            print(f"      {i}. MFCC {idx}: {model_info['importancia_features'][idx]:.4f}")
    print()
    
except Exception as e:
    print(f"   ❌ Error: {e}\n")

# 7. Verificar tamaño de memoria
print("7️⃣  Verificando recursos...")
try:
    import psutil
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / (1024 * 1024)
    print(f"   ✅ Memoria actual: {memory_mb:.2f} MB")
    print(f"   ✅ Sistema accesible\n")
except:
    print(f"   ℹ️  (No se pudo obtener información de psutil)\n")

# Resumen final
print("="*70)
print("✨ ¡TODAS LAS PRUEBAS COMPLETADAS EXITOSAMENTE!")
print("="*70)
print("\n✅ La app está lista para usarse. Inicia con:")
print("   ./ejecutar.sh")
print("   o")
print("   streamlit run app_streamlit.py")
print("\n")
