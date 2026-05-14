# 🐋 Clasificador de Cetáceos por Sonido

Una aplicación web interactiva con Streamlit que identifica especies de cetáceos (ballenas y delfines) analizando sus sonidos mediante inteligencia artificial.

## 🎯 Características

- **Identificación de Cetáceos**: Carga un audio y obtén la predicción automática de qué cetáceo es
- **Confianza de Predicción**: Muestra el porcentaje de confianza en la predicción
- **Información Detallada**: Cada predicción incluye descripción científica y características del cetáceo
- **Análisis Completo**: Visualiza las probabilidades de todas las especies clasificadas
- **Interfaz Amigable**: Diseño intuitivo y responsivo con Streamlit

## 📊 Especies Soportadas

El modelo puede clasificar las siguientes especies de cetáceos:

1. 🐋 **Ballena Azul** - Balaenoptera musculus
2. 🐋 **Ballena de Aleta** - Balaenoptera physalus
3. 🐋 **Ballena Jorobada** - Megaptera novaeangliae
4. 🐋 **Orca** - Orcinus orca
5. 🐋 **Ballena Enana** - Balaenoptera acutorostrata
6. 🐋 **Ballena Franca del Atlántico Norte** - Eubalaena glacialis
7. 🐋 **Ballena Pico del Norte** - Hyperoodon ampullatus
8. 🐋 **Ballena Piloto** - Globicephala spp
9. 🐋 **Ballena Sei** - Balaenoptera borealis
10. 🐋 **Cachalote** - Physeter macrocephalus

## 🚀 Instalación y Uso

### Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Instalación

1. **Clona o descarga el proyecto**:
```bash
cd /home/azureuser/cloudfiles/code
```

2. **Instala las dependencias**:
```bash
pip install -r requirements.txt
```

### Ejecución

Ejecuta la siguiente orden en la terminal:

```bash
streamlit run app_streamlit.py
```

La aplicación se abrirá en tu navegador (por defecto en `http://localhost:8501`)

## 📝 Cómo Usar

1. **Carga un Audio**: Selecciona un archivo de audio (.wav, .mp3, .flac, .ogg)
2. **Reproduce el Audio**: Escucha el audio cargado con el reproductor integrado
3. **Obtén la Predicción**: El sistema analiza automáticamente el audio
4. **Visualiza los Resultados**: 
   - Especie identificada con confianza
   - Descripción completa del cetáceo
   - Gráfico con probabilidades de todas las especies

## 🛠️ Tecnología Utilizada

- **Streamlit**: Framework web de Python para interfaces interactivas
- **Librosa**: Librería para análisis de audio
- **scikit-learn**: Machine Learning (modelo Random Forest)
- **NumPy/Pandas**: Procesamiento de datos numéricos

## 📊 Modelo de Machine Learning

El clasificador utiliza un modelo **Random Forest** entrenado con:
- Características de audio: MFCC (Mel-Frequency Cepstral Coefficients)
- 13 características acústicas por cada audio
- Datos de entrenamiento de múltiples sonidos de cetáceos

## ⚙️ Requisitos de Sistema

- Espacio en disco: ~100 MB (incluyendo dependencias)
- RAM recomendada: 2 GB
- Navegador web moderno

## 📁 Estructura de Archivos

```
/home/azureuser/cloudfiles/code/
├── app_streamlit.py                 # Aplicación Streamlit principal
├── requirements.txt                 # Dependencias del proyecto
├── README.md                        # Este archivo
└── Users/alvaro.lopezredondo.3431/
    ├── clasificador_ballenas.pkl    # Modelo entrenado
    ├── resultados_ballenas_mapa.csv # Datos de entrenamiento
    └── extraer.ipynb               # Notebook del proceso
```

## 🔧 Solución de Problemas

### Error: "Modelo no encontrado"
- Verifica que el archivo `clasificador_ballenas.pkl` está en la ruta correcta
- Por defecto: `/home/azureuser/cloudfiles/code/Users/alvaro.lopezredondo.3431/`

### Error: "No se puede cargar el audio"
- Asegúrate que el archivo está en un formato soportado (.wav, .mp3, .flac, .ogg)
- Intenta con un archivo de menor tamaño si hay problemas de memoria

### La predicción es incorrecta
- La precisión del modelo depende de:
  - Calidad del audio
  - Claridad de la vocalización
  - Similitud con los audios de entrenamiento

## 📚 Referencias

- [Librosa Documentation](https://librosa.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [scikit-learn RandomForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

## 👨‍💻 Autor

Proyecto desarrollado como parte del análisis de cetáceos mediante aprendizaje automático.

## 📄 Licencia

Este proyecto se proporciona tal cual para fines educativos y de investigación.

---

**¿Necesitas ayuda?** Asegúrate de que todas las dependencias están instaladas con `pip install -r requirements.txt`
