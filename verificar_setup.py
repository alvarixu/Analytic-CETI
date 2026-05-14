#!/usr/bin/env python3
"""
Script de prueba para verificar que el clasificador está correctamente configurado
"""

import sys
import os

print("=" * 60)
print("🐋 Verificación del Clasificador de Cetáceos")
print("=" * 60)

# 1. Verificar importaciones
print("\n1️⃣  Verificando importaciones...")
try:
    import streamlit
    print("   ✅ Streamlit OK")
except ImportError:
    print("   ❌ Streamlit no instalado")
    sys.exit(1)

try:
    import librosa
    print("   ✅ Librosa OK")
except ImportError:
    print("   ❌ Librosa no instalada")
    sys.exit(1)

try:
    import joblib
    print("   ✅ Joblib OK")
except ImportError:
    print("   ❌ Joblib no instalado")
    sys.exit(1)

try:
    import sklearn
    print("   ✅ scikit-learn OK")
except ImportError:
    print("   ❌ scikit-learn no instalada")
    sys.exit(1)

try:
    import numpy
    print("   ✅ NumPy OK")
except ImportError:
    print("   ❌ NumPy no instalado")
    sys.exit(1)

try:
    import pandas
    print("   ✅ Pandas OK")
except ImportError:
    print("   ❌ Pandas no instalado")
    sys.exit(1)

# 2. Verificar archivos
print("\n2️⃣  Verificando archivos necesarios...")
model_path = "/home/azureuser/cloudfiles/code/Users/alvaro.lopezredondo.3431/clasificador_ballenas.pkl"
app_path = "/home/azureuser/cloudfiles/code/app_streamlit.py"

if os.path.exists(model_path):
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"   ✅ Modelo encontrado ({size_mb:.2f} MB)")
else:
    print(f"   ❌ Modelo NO encontrado en {model_path}")
    sys.exit(1)

if os.path.exists(app_path):
    print(f"   ✅ App Streamlit encontrada")
else:
    print(f"   ❌ App Streamlit NO encontrada en {app_path}")
    sys.exit(1)

# 3. Cargar y probar el modelo
print("\n3️⃣  Cargando el modelo...")
try:
    import joblib
    model = joblib.load(model_path)
    print(f"   ✅ Modelo cargado correctamente")
    print(f"   📊 Tipo: {type(model).__name__}")
    print(f"   🏷️  Clases soportadas: {len(model.classes_)}")
    print(f"   📝 Clases:")
    for i, clase in enumerate(model.classes_, 1):
        print(f"      {i}. {clase}")
except Exception as e:
    print(f"   ❌ Error al cargar el modelo: {e}")
    sys.exit(1)

# 4. Verificar características del modelo
print("\n4️⃣  Verificando características del modelo...")
try:
    n_features = model.n_features_in_
    print(f"   ✅ Número de características esperadas: {n_features}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# 5. Prueba con datos aleatorios
print("\n5️⃣  Prueba de predicción...")
try:
    import numpy as np
    # Crear features aleatorios (13 features MFCC)
    test_features = np.random.randn(1, 13)
    prediction = model.predict(test_features)
    probabilities = model.predict_proba(test_features)
    
    print(f"   ✅ Predicción de prueba: {prediction[0]}")
    print(f"   ✅ Confianza: {np.max(probabilities) * 100:.2f}%")
except Exception as e:
    print(f"   ❌ Error en predicción: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✨ ¡Todo está correctamente configurado!")
print("=" * 60)
print("\nPara ejecutar la app, usa:")
print("   streamlit run /home/azureuser/cloudfiles/code/app_streamlit.py")
print("=" * 60)
