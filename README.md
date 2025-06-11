# 🚗 RiskRadar System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![YOLO](https://img.shields.io/badge/YOLO-v8-yellow.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

**Sistema avanzado de análisis de riesgo vehicular en tiempo real usando Computer Vision y Deep Learning**

---

## 📋 Descripción

RiskRadar es un sistema inteligente que analiza videos de tráfico para detectar y evaluar riesgos de colisión en tiempo real. Utiliza múltiples modelos de IA para detectar vehículos y objetos, estimar profundidad, y generar mapas de calor de riesgo en una zona de interés definida por el usuario.

### ✨ Características Principales

- 🎯 **Detección Multi-Modelo**: Combina YOLO personalizado y COCO para máxima precisión
- 🌡️ **Mapas de Calor de Riesgo**: Visualización en tiempo real de zonas peligrosas
- 📊 **Estimación de Profundidad**: Usa MiDaS para análisis espacial 3D
- 🔄 **Seguimiento Optimizado**: Sistema híbrido detección/tracking para mejor rendimiento
- 📈 **Análisis Estadístico**: Reportes detallados y visualizaciones
- ⚡ **Procesamiento Eficiente**: Optimizado para videos de alta resolución

## 🛠️ Instalación

### Prerrequisitos

```bash
Python >= 3.8
CUDA >= 11.0 (recomendado para GPU)
```

### Instalación de Dependencias

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/riskradar-system.git
cd riskradar-system

# Instalar dependencias principales
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics opencv-python numpy matplotlib

# Dependencias adicionales
pip install timm  # Para MiDaS
```

### Modelos Requeridos

1. **Modelo YOLO personalizado**: Coloca tu modelo `best.pt` en el directorio del proyecto
2. **MiDaS**: Se descarga automáticamente desde PyTorch Hub
3. **YOLOv8n**: Se descarga automáticamente

## 🚀 Uso Rápido

### Configuración Básica

```python
config = {
    'MODEL_PATH_VEHICLES': 'path/to/your/best.pt',
    'VIDEO_INPUT_PATH': 'path/to/your/video.mp4',
    'OUTPUT_DIR': 'results',
    'YOLO_CONFIDENCE_THRESHOLD': 0.40,
    'PROCESSING_RESOLUTION': (960, 540)
}
```

### Ejecutar el Sistema

```bash
python riskradar_system.py
```

## ⚙️ Configuración Avanzada

### Parámetros Principales

| Parámetro | Descripción | Valor por Defecto |
|-----------|-------------|-------------------|
| `PROCESS_EVERY_N_FRAMES` | Frecuencia de detección completa | `5` |
| `YOLO_CONFIDENCE_THRESHOLD` | Umbral de confianza YOLO | `0.40` |
| `NMS_IOU_THRESHOLD` | Umbral IoU para NMS | `0.6` |
| `CONE_TOP_WIDTH_FACTOR` | Ancho del cono de riesgo | `0.8` |
| `HEATMAP_DECAY_RATE` | Tasa de decaimiento del mapa de calor | `0.85` |

### Clases Detectadas

**Vehículos (Modelo Personalizado):**
- `car`, `bus`, `truck`, `van`, `motorbike`, `threewheel`

**COCO Classes:**
- `person` (ID: 0)
- `bicycle` (ID: 1) 
- `dog` (ID: 16)

## 📊 Resultados y Reportes

El sistema genera automáticamente:

### Archivos de Salida
- 🎥 `output_risk_radar.mp4` - Video procesado con visualizaciones
- 📄 `detections_raw.json` - Datos de detección sin procesar
- 📊 `detections_raw.csv` - Detecciones en formato CSV
- 📈 `analysis_charts.png` - Gráficos de análisis
- 📋 `statistics_report.json` - Estadísticas detalladas
- 📝 `summary_report.txt` - Resumen ejecutivo

### Visualizaciones Incluidas
- Distribución de detecciones por clase
- Evolución del nivel de calor/riesgo
- Tiempo en cada nivel de riesgo
- Tiempos de procesamiento por frame

## 🏗️ Arquitectura del Sistema

```
RiskRadarSystem
├── Detección Multi-Modelo
│   ├── YOLO Personalizado (Vehículos)
│   └── YOLOv8n COCO (Personas, Bicicletas, etc.)
├── Estimación de Profundidad (MiDaS)
├── Seguimiento de Objetos (CSRT)
├── Análisis de Riesgo
│   ├── Mapa de Calor Dinámico
│   └── Evaluación por Zona de Interés
└── Generación de Reportes
```

## 🎯 Algoritmo de Riesgo

El sistema evalúa el riesgo basándose en:

1. **Factor de Profundidad**: Objetos más cercanos = mayor riesgo
2. **Factor de Posición**: Objetos en parte inferior del frame = mayor riesgo
3. **Tipo de Objeto**: Cada clase tiene un peso específico de riesgo
4. **Zona de Interés**: Solo objetos dentro del cono de análisis

```python
heat_value = base_heat × depth_factor × position_factor
```

## 📈 Rendimiento

- ⚡ **Optimización GPU**: Soporte completo CUDA
- 🔄 **Tracking Híbrido**: Reduce carga computacional 80%
- 📱 **Resolución Adaptativa**: Procesa a resolución optimizada
- 🎯 **NMS Inteligente**: Elimina detecciones duplicadas

## 🤝 Contribuir

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/nueva-caracteristica`)
3. Commit tus cambios (`git commit -m 'Agregar nueva característica'`)
4. Push a la rama (`git push origin feature/nueva-caracteristica`)
5. Abre un Pull Request

## 📝 Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## 🙏 Reconocimientos

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - Framework de detección
- [MiDaS](https://github.com/isl-org/MiDaS) - Estimación de profundidad
- [OpenCV](https://opencv.org/) - Procesamiento de imágenes


<div align="center">

**⭐ Si este proyecto te fue útil, considera darle una estrella ⭐**

![GitHub stars](https://img.shields.io/github/stars/tu-usuario/riskradar-system?style=social)
![GitHub forks](https://img.shields.io/github/forks/tu-usuario/riskradar-system?style=social)

</div>
