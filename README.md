# Memoria — Cuantificación de gases de tronadura desde video de dron

Código de la memoria de título (UTFSM). Estima, a partir de un video de dron, la columna de gas de una tronadura en la mina Chuquicamata. El proceso tiene cinco fases:

| Fase | Contenido | Salida principal |
|---|---|---|
| 0 | Preprocesamiento: instante de detonación, pose de la cámara, modelo de fondo y georreferenciación | `geometria.json`, `georreferencia.json` |
| 1 | Segmentación del gas con CLIPSeg + SAM 2 | `mascaras_gas.pkl` |
| 2 | Reconstrucción 3D por *space carving* (cámara + sol), altura por sombra y exportación a Blender | `serie_columna.csv`, `gas_secuencia.glb` |
| 3 | Viento horizontal por flujo óptico sobre el plano de advección y contraste con estaciones | `viento_mundo.csv` |
| 4 | Dispersión (σ) y concentración normalizada χ = C/Q con un modelo de *puff* | `parametros_dispersion.json` |

## Estructura

```
├── Codigo.ipynb                 pipeline completo (fases 0–4)
├── funciones/                   módulos funciones_F0..F4 (import funciones as FN)
├── datos/                       insumos
│   ├── DEM_mina.tif
│   ├── anotaciones_20250815165143.json   trazado manual de la columna (F1-9)
│   └── estaciones/              DMC Chorrillos, SINCA P. Vergara Keller
├── Validacion_Sintetica/        pipeline contra una simulación con verdad-terreno
│   ├── validacion_completa.ipynb
│   ├── funciones/
│   └── datos/config_15ago.json
└── requirements.txt
```

La carpeta `resultados/` se crea automáticamente al correr el notebook.

## Instalación

Probado con Python 3.13 en Windows, solo con CPU.

```bash
pip install -r requirements.txt
```

`requirements.txt` instala SAM 2 desde su repositorio (`facebookresearch/sam2`).

## Datos que no están en el repositorio

Por su tamaño, los siguientes archivos no están en el repositorio. Deben dejarse en `datos/` (y en `Validacion_Sintetica/datos/` para la validación):

| Archivo | Origen |
|---|---|
| `DJI_20250815165143_0003_V.MP4` | Video del dron (4K) de la tronadura del 15-08-2025. No es público. |
| `sam2_hiera_tiny.pt` | Pesos de SAM 2: https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt |
| `clipseg-rd64-refined/` | Se descarga sola desde Hugging Face (`CIDAS/clipseg-rd64-refined`) en la primera ejecución y queda guardada en `datos/`. |
| `satelital_utm.tif` | Ortofoto exportada desde QGIS con la capa XYZ *Google Satellite*: 1 m/px, EPSG:32719, E 506 500–513 668 m, N 7 532 356–7 538 500 m. |
| `nube_volumetrica.avi`, `nube_difusa.avi`, `fondo_nube_roja_sin_sombra.avi` | Renders de la simulación usados en la validación. No son públicos. |

Para acceder a los videos, contactar al autor.

## Uso

### Pipeline

1. Abrir `Codigo.ipynb` desde la raíz del repositorio.
2. Fijar `VIDEO_NOMBRE` (celda *VIDEO A PROCESAR*) con el nombre del MP4 en `datos/`.
3. Correr las celdas en orden. Cada fase guarda sus resultados en `resultados/<n>_<fase>/`, y la fase siguiente los lee desde el disco.

En la primera ejecución, la Fase 0 abre una ventana para marcar con un clic el centro de la tronadura (pivote). El pivote queda en `resultados/0_preprocesamiento/pivote_manual_<video>.pkl` y no se vuelve a pedir.

Las celdas de validación de la Fase 1 (métrica interna de SAM 2, consistencia y videos de variantes) están desactivadas por defecto con banderas `False`, porque son costosas: la de consistencia tarda unas 10 h en CPU. El pipeline sin ellas tarda alrededor de 70 min.

### Blender

La Fase 2 escribe en `resultados/2_reconstruccion3d/` los archivos `gas_secuencia.glb`, `terreno.glb`, `camara_dron.json` y `blender_animar_gas.py`. En Blender, ir a *Scripting → Open* y abrir `blender_animar_gas.py`, luego *Run Script*. El script busca los archivos en su propia carpeta.

### Validación sintética

Abrir `Validacion_Sintetica/validacion_completa.ipynb`:

1. Con `RUN = "volumetrica"` en VAL-0, correr hasta el final de la Fase 4.
2. Repetir con `RUN = "difusa"`.
3. Correr la sección de validación, que compara ambas corridas con la verdad-terreno.

La validación usa el mismo DEM y la misma ortofoto que el pipeline, que deben copiarse a `Validacion_Sintetica/datos/`.
