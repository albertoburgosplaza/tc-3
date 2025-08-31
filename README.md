# Local Object Detector

Un servicio local conteinerizado en Python para detección de objetos (persona y coche) usando YOLOv8. El servicio proporciona una API HTTP que recibe imágenes y devuelve detecciones en formato JSON.

## Características

- ✅ Detección de **personas** y **coches** usando YOLOv8
- ✅ API REST con FastAPI
- ✅ Conteinerizado con Docker 
- ✅ Optimizado para CPU (sin dependencia de GPU)
- ✅ Validación robusta de entrada
- ✅ Logs detallados y métricas de rendimiento
- ✅ Manejo completo de errores

## Arquitectura

```
Local Object Detector
├── FastAPI (servidor web)
├── YOLOv8 (modelo de detección)
├── PIL + OpenCV (procesamiento de imágenes)
└── Docker (conteinerización)
```

## Instalación y Ejecución

### Opción 1: Docker Compose (Recomendado para Producción)

#### Ejecutar el servicio
```bash
# Construir e iniciar el servicio
docker-compose up -d

# Ver logs en tiempo real
docker-compose logs -f

# Parar el servicio
docker-compose down
```

### Opción 2: Docker Manual

#### Construir la imagen
```bash
docker build -t local-detector:latest .
```

#### Ejecutar el contenedor
```bash
docker run --rm -p 8000:8000 local-detector:latest
```

El servicio estará disponible en: http://localhost:8000

### Opción 3: Instalación local

#### Prerrequisitos
- Python 3.11+
- pip

#### Instalación
```bash
# Clonar el repositorio
git clone <repository-url>
cd turingchallenge-reto-3

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar el servicio
python main.py
```

## API Reference

### Health Check

Verifica que el servicio esté funcionando correctamente.

**Endpoint:** `GET /health`

**Respuesta:**
```json
{
  "status": "healthy",
  "service": "Local Object Detector",
  "version": "1.0.0"
}
```

### Detección de Objetos

Realiza detección de personas y coches en una imagen.

**Endpoint:** `POST /infer`

**Content-Type:** `multipart/form-data`

#### Parámetros

| Parámetro | Tipo | Requerido | Por defecto | Descripción |
|-----------|------|-----------|-------------|-------------|
| `image` | File | ✅ Sí | - | Archivo de imagen (JPEG/PNG, máx. 10MB) |
| `conf` | float | ❌ No | 0.25 | Umbral de confianza (0.0-1.0) |
| `iou` | float | ❌ No | 0.45 | Umbral IoU para supresión no máxima (0.0-1.0) |
| `max_detections` | int | ❌ No | 300 | Número máximo de detecciones (1-1000) |

#### Descripción de parámetros

- **conf (confidence)**: Umbral de confianza mínimo para considerar una detección válida. Valores más altos = menos detecciones pero más precisas.
- **iou (Intersection over Union)**: Umbral para eliminar detecciones duplicadas. Valores más altos = permite más detecciones solapadas.
- **max_detections**: Límite máximo de objetos a detectar en la imagen.

#### Formato de respuesta JSON

```json
{
  "model": "yolov8n.pt",
  "time_ms": 150.25,
  "image": {
    "width": 1920,
    "height": 1080
  },
  "detections": [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.85,
      "bbox_xyxy": [100, 200, 300, 500],
      "bbox_xywh": [200, 350, 200, 300],
      "bbox_norm_xyxy": [0.052, 0.185, 0.156, 0.463]
    }
  ]
}
```

#### Descripción de campos de respuesta

- **model**: Nombre del modelo YOLO utilizado
- **time_ms**: Tiempo de inferencia en milisegundos
- **image**: Dimensiones de la imagen procesada
- **detections**: Lista de objetos detectados
  - **class_id**: ID numérico de la clase (0=person, 2=car)
  - **class_name**: Nombre legible de la clase
  - **confidence**: Confianza de la detección (0.0-1.0)
  - **bbox_xyxy**: Bounding box formato [x1, y1, x2, y2] en píxeles
  - **bbox_xywh**: Bounding box formato [x_center, y_center, width, height] en píxeles
  - **bbox_norm_xyxy**: Bounding box normalizado (valores 0.0-1.0)

## Ejemplos de Uso

### Ejemplo con curl

#### Test básico de salud
```bash
curl http://localhost:8000/health
```

#### Detección básica
```bash
curl -X POST \
  http://localhost:8000/infer \
  -F "image=@test_person.jpg"
```

#### Detección con parámetros personalizados
```bash
curl -X POST \
  http://localhost:8000/infer \
  -F "image=@test_person.jpg" \
  -F "conf=0.5" \
  -F "iou=0.3" \
  -F "max_detections=100"
```

#### Ejemplo con archivo remoto
```bash
# Descargar imagen de ejemplo
wget -O example.jpg "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d"

# Realizar detección
curl -X POST \
  http://localhost:8000/infer \
  -F "image=@example.jpg" \
  -F "conf=0.3"
```

### Ejemplo con Python

#### Instalación de dependencias para el cliente
```bash
pip install requests pillow
```

#### Script básico de Python
```python
import requests
import json

def detect_objects(image_path, conf=0.25, iou=0.45, max_detections=300):
    """
    Realiza detección de objetos usando el servicio local
    
    Args:
        image_path (str): Ruta al archivo de imagen
        conf (float): Umbral de confianza
        iou (float): Umbral IoU
        max_detections (int): Número máximo de detecciones
    
    Returns:
        dict: Resultado de la detección
    """
    url = "http://localhost:8000/infer"
    
    # Preparar archivos y datos
    with open(image_path, 'rb') as img_file:
        files = {'image': img_file}
        data = {
            'conf': conf,
            'iou': iou,
            'max_detections': max_detections
        }
        
        # Realizar petición
        response = requests.post(url, files=files, data=data)
    
    # Verificar respuesta
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error {response.status_code}: {response.text}")

# Ejemplo de uso
if __name__ == "__main__":
    try:
        # Verificar salud del servicio
        health = requests.get("http://localhost:8000/health")
        print("Servicio disponible:", health.json())
        
        # Realizar detección
        result = detect_objects("test_person.jpg", conf=0.5)
        
        print(f"\nDetección completada en {result['time_ms']:.2f}ms")
        print(f"Imagen: {result['image']['width']}x{result['image']['height']} píxeles")
        print(f"Detecciones encontradas: {len(result['detections'])}")
        
        # Mostrar detecciones
        for i, det in enumerate(result['detections']):
            print(f"\nDetección {i+1}:")
            print(f"  Clase: {det['class_name']} (ID: {det['class_id']})")
            print(f"  Confianza: {det['confidence']:.3f}")
            print(f"  Posición: {det['bbox_xyxy']}")
            
    except Exception as e:
        print(f"Error: {e}")
```

#### Script avanzado con procesamiento de múltiples imágenes
```python
import requests
import json
from pathlib import Path
import time

class ObjectDetectorClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        
    def health_check(self):
        """Verificar estado del servicio"""
        response = requests.get(f"{self.base_url}/health")
        return response.json() if response.status_code == 200 else None
        
    def detect(self, image_path, **kwargs):
        """Detectar objetos en una imagen"""
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(f"{self.base_url}/infer", files=files, data=kwargs)
            
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Error {response.status_code}: {response.text}")
    
    def batch_detect(self, image_folder, pattern="*.jpg", **detection_kwargs):
        """Procesar múltiples imágenes"""
        image_folder = Path(image_folder)
        results = []
        
        for image_path in image_folder.glob(pattern):
            print(f"Procesando {image_path.name}...")
            
            try:
                start_time = time.time()
                result = self.detect(image_path, **detection_kwargs)
                end_time = time.time()
                
                result['filename'] = image_path.name
                result['processing_time'] = end_time - start_time
                results.append(result)
                
                print(f"  ✅ {len(result['detections'])} detecciones en {result['time_ms']:.1f}ms")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                
        return results

# Ejemplo de uso
if __name__ == "__main__":
    client = ObjectDetectorClient()
    
    # Verificar servicio
    if not client.health_check():
        print("❌ Servicio no disponible")
        exit(1)
    
    print("✅ Servicio disponible")
    
    # Procesar múltiples imágenes
    results = client.batch_detect("./images", "*.jpg", conf=0.3, max_detections=50)
    
    # Estadísticas generales
    total_images = len(results)
    total_detections = sum(len(r['detections']) for r in results)
    avg_time = sum(r['time_ms'] for r in results) / total_images if total_images > 0 else 0
    
    print(f"\n📊 Estadísticas:")
    print(f"  Imágenes procesadas: {total_images}")
    print(f"  Total detecciones: {total_detections}")
    print(f"  Tiempo promedio: {avg_time:.1f}ms")
```

## Códigos de Estado HTTP

| Código | Descripción | Causa común |
|--------|-------------|-------------|
| 200 | OK | Detección exitosa |
| 400 | Bad Request | Formato de imagen no soportado o imagen corrupta |
| 413 | Payload Too Large | Imagen mayor a 10MB |
| 422 | Unprocessable Entity | Parámetros inválidos (conf, iou, max_detections fuera de rango) |
| 500 | Internal Server Error | Error interno del modelo o servidor |

## Logs y Monitoreo

El servicio genera logs detallados que incluyen:

- **Métricas de rendimiento**: Tiempo de inferencia, tamaño de imagen, número de detecciones
- **Información de peticiones**: Archivos procesados, parámetros utilizados
- **Errores y advertencias**: Problemas de validación, errores internos

Ejemplo de log de rendimiento:
```
2024-08-30 14:30:15 - app.app - INFO - PERFORMANCE_METRICS - file='test.jpg' image_size=1920x1080 file_size_bytes=245760 parameters=conf:0.25,iou:0.45,max_det:300 inference_time_ms=150.25 total_time_ms=165.30 processing_overhead_ms=15.05 detections_total=3 detections_by_class={'person': 2, 'car': 1}
```

## Resolución de Problemas

### Problemas Comunes

#### El contenedor no inicia
```bash
# Verificar logs del contenedor
docker logs <container-id>

# Verificar que el puerto esté libre
lsof -i :8000  # Linux/Mac
netstat -ano | findstr :8000  # Windows
```

#### Error "Model file not found"
```bash
# Verificar que existe el archivo del modelo
ls -la models/

# Reconstruir la imagen Docker
docker build --no-cache -t local-detector:latest .
```

#### Respuestas lentas
- Reducir `max_detections` para imágenes con muchos objetos
- Usar imágenes de menor resolución cuando sea posible
- Aumentar `conf` para reducir detecciones de baja confianza

#### Demasiadas detecciones falsas
- Aumentar el parámetro `conf` (ej: 0.5 o 0.7)
- Ajustar `iou` para eliminar detecciones solapadas

## Rendimiento

### Métricas típicas (CPU Intel i5, imagen 720p)
- **Tiempo de inferencia**: 150-300ms
- **Throughput**: ~10-15 req/s
- **Memoria**: ~2-3GB (incluyendo modelo)

### Recomendaciones de optimización
- Usar imágenes de menor resolución cuando sea posible
- Ajustar `max_detections` según las necesidades
- Considerar usar `yolov8s.pt` si se requiere mayor precisión

## Desarrollo

### Estructura del proyecto
```
turingchallenge-reto-3/
├── app/
│   ├── __init__.py
│   ├── app.py          # FastAPI app principal
│   └── config.py       # Configuración
├── models/             # Modelos YOLO
├── tests/             # Tests unitarios y de integración
│   ├── conftest.py     # Fixtures compartidas
│   └── test_basic_setup.py
├── Dockerfile         # Imagen Docker
├── docker-compose.yml # Configuración de producción
├── main.py           # Punto de entrada
├── requirements.txt  # Dependencias Python
├── run_tests_complete.py # Script completo de testing
└── README.md         # Esta documentación
```

### Comandos de Testing

```bash
# Ejecutar todos los tests con cobertura
python run_tests_complete.py

# Modo rápido (sin tests de rendimiento)
python run_tests_complete.py --quick

# Solo cobertura (unit + integration)
python run_tests_complete.py --coverage-only

# Tests específicos por categoría
pytest tests/unit/ -v -m unit
pytest tests/integration/ -v -m integration
pytest tests/performance/ -v -m performance

# Test individual
pytest tests/test_basic_setup.py::test_function_name -v
```

### Variables de entorno

| Variable | Descripción | Por defecto |
|----------|-------------|-------------|
| `MODEL_PATH` | Ruta al modelo YOLO | `models/yolov8n.pt` |
| `DEFAULT_CONF` | Confianza por defecto | `0.25` |
| `DEFAULT_IOU` | IoU por defecto | `0.45` |
| `DEFAULT_MAX_DETECTIONS` | Detecciones máximas por defecto | `300` |

## Licencia

Este proyecto utiliza las siguientes dependencias principales:
- **YOLOv8 (Ultralytics)**: AGPL-3.0
- **FastAPI**: MIT
- **PyTorch**: BSD-3-Clause

Revisar las licencias individuales para uso comercial.