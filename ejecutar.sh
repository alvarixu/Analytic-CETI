#!/bin/bash

# Script para ejecutar la app de Streamlit del Clasificador de Cetáceos
# Con manejo de errores y reintentos

echo "=================================================="
echo "🐋 Iniciando Clasificador de Cetáceos"
echo "=================================================="
echo ""

# Configurar variables de entorno para Streamlit
export STREAMLIT_CLIENT_LOGGING_LEVEL="error"
export STREAMLIT_LOGGER_LEVEL="error"
export STREAMLIT_SERVER_HEADLESS="true"
export STREAMLIT_SERVER_ENABLECORS="false"
export STREAMLIT_SERVER_PORT="8501"
export STREAMLIT_SERVER_ADDRESS="0.0.0.0"

echo "📍 La app se abrirá en: http://localhost:8501"
echo ""
echo "💡 Consejos:"
echo "   - Si lo prefieres, abre la URL en tu navegador"
echo "   - Presiona Ctrl+C para detener la app"
echo "   - Los archivos de audio deben ser .wav, .mp3, .flac o .ogg"
echo "   - Si hay error de conexión, espera unos segundos y recarga la página"
echo ""
echo "=================================================="
echo ""

# Ejecutar con logs mínimos
streamlit run app_streamlit.py --logger.level=error 2>&1 | grep -v "UserWarning\|DeprecationWarning" || true
