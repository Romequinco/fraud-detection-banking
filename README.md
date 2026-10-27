# fraud-detection-banking · Detección de fraude en transacciones bancarias

Práctica en grupo del Máster MIAX (módulo de Detección de Anomalías): construir un sistema que
asigne a cada transacción bancaria un **score de fraude**.

## Grupo

- Oscar Romero Quincoces
- Fernando Dapena Tauste
- Daniel Garcia Lopez

## Enunciado (resumen)

- **Objetivo:** detectar transacciones fraudulentas maximizando el **AUPR** (área bajo la curva
  Precision-Recall, `sklearn.metrics.average_precision_score`) en producción.
- **Entregables** (aula virtual, hasta las **18:30 del 29 de octubre**):
  1. Un notebook `.ipynb` con los nombres del grupo, el código, la justificación de los desarrollos
     y una reflexión final → `notebooks/99_ENTREGA.ipynb`.
  2. Un CSV con **una sola columna**: el score de cada fila del fichero de producción
     (mayor score = mayor probabilidad de fraude) → `data/output/scores_produccion.csv`.
- **Evaluación:** resultados (AUPR en producción) 75 % · notebook 25 %.
- Enunciado completo: [`docs/enunciado.pdf`](docs/enunciado.pdf).

## Datos

| Fichero | Filas | Columnas | Etiqueta |
|---|---|---|---|
| `data/credit_card_construction.csv` | 170.884 | `X1..X20`, `Amount`, `Class` | sí (295 fraudes, 0,17 %) |
| `data/credit_card_production.csv` | 113.923 | `X1..X20`, `Amount` (`Class` vacía) | no |

Variables anonimizadas, sin nulos. Los CSV están versionados: el repositorio es plug-and-play.

## Estructura

```
├── README.md              # este fichero (hace también de índice)
├── requirements.txt
├── data/
│   ├── credit_card_construction.csv
│   ├── credit_card_production.csv
│   └── output/            # 01_dataset_preparado.csv (lo leen 02–04), NN_*.csv y scores_produccion.csv (entregable)
├── docs/
│   ├── enunciado.pdf
│   ├── DECISIONES.md      # registro de decisiones D-X.n
│   ├── teoria/            # síntesis del material de clase (extraído de _fuentes/)
│   └── _fuentes/          # material bruto del profesor (NO versionado)
└── notebooks/
    ├── 01_EDA_y_preprocesado.ipynb
    ├── 02_baselines_no_supervisados.ipynb
    ├── 03_supervisados.ipynb
    ├── 04_hibrido_y_ensemble.ipynb
    └── 99_ENTREGA.ipynb   # notebook único de entrega (autocontenido)
```

## Documentación

- [`docs/DECISIONES.md`](docs/DECISIONES.md): decisiones del proyecto.
- `docs/teoria/` (síntesis completa del material de clase, con cita de la fuente exacta en cada concepto):
  - `00_fundamentos.md`: definiciones, tipos de anomalía (local, global, dependiente, agrupada), supervisión, ruido, data shift, OOD.
  - `01_metodos_clasicos.md`: distancias (Lp, coseno, Mahalanobis, kNN, LOF, DBSCAN), probabilidad (gaussiana, GMM, KDE), información (entropía, MI, KL, RBIG), reconstrucción y modelos (regresión, PCA, OCSVM, IsolationForest).
  - `02_deep.md`: autoencoders, AE convolucional, discriminador GAN, flows y autorregresivos.
  - `03_evaluacion_y_herramientas.md`: AUPR, desbalanceo/SMOTE, comparativa de métodos, explicabilidad de outliers, sklearn/PyOD.
  - `notebook_partida.md`: formato exacto de la entrega, dataset, notebook de partida, línea base (AUPR 0,567), estilo de código y recomendaciones consolidadas del profesor. **Empezad por aquí.**
- `docs/_fuentes/` (local, no versionado): enunciado original, notebook de partida, transcripción de la clase, slides (`Deteccion_de_anomalias.pdf`, `AD_algoritmos_deep.pdf`), papers y libro, notebooks `scripts_classic/` y `scripts_deep/` (versión `_clase` = solución) y zips originales. El detalle está en `_fuentes/INVENTARIO.md`.

## Cómo trabajar

Entorno local en Windows (Python 3.13, CPU). El kernel del proyecto se llama **Fraude (.venv)**.
Para recrearlo con las versiones verificadas:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-windows.lock.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name fraud-detection-banking --display-name "Fraude (.venv)"
```

Para abrir los notebooks con el entorno ya creado:

```powershell
.\.venv\Scripts\python.exe -m jupyterlab notebooks/
```

Seleccionar **Fraude (.venv)** como kernel del notebook. En VS Code también se puede elegir
directamente `.venv\Scripts\python.exe`. No hace falta activar el entorno para usar los comandos
anteriores. `.venv/` está excluido de Git; `requirements.txt` mantiene la lista general de dependencias.

- **01** genera `data/output/01_dataset_preparado.csv`: split fijo train/test (el del profesor),
  `fold` de validación (StratifiedGroupKFold por duplicados) y variables sin estado.
- **02–04 y 99** empiezan con la misma **celda común v2 (no modificar)**: leen ese CSV y exponen
  `X_train/X_test/X_prod`, `cv_splits`, `make_pipe()` (escalado dentro del fold), `cv_aupr()`,
  `report()` (AUPR de test con y sin duplicados), `plot_pr()` y `save_submission()`. Así los AUPR de
  todos los miembros son comparables.
- Cada notebook guarda sus tablas en `data/output/NN_*.csv`.
- **03** tarda unos 10 min (Random Forest y MLP son lo más lento) y deja `03_metricas_cv.csv`,
  `03_metricas_test.csv`, `03_scores.csv` (scores para el 04, mismo convenio que `02_scores.csv`) y
  `03_scores_produccion_supervisado.csv` (respaldo supervisado).
- **02** tarda unos 19 min de principio a fin (barridos de GMM, LOF y un autoencoder en PyTorch) y deja
  `02_metricas_cv.csv`, `02_metricas_test.csv`, `02_scores.csv` (scores para el 04) y
  `02_scores_produccion_no_supervisado.csv` (entregable de respaldo).
- Las decisiones relevantes se registran en `docs/DECISIONES.md`.
- `99_ENTREGA.ipynb` consolida al final el código real de `01`–`04` y debe poder ejecutarse sin el
  resto del repositorio.

## Resultados hasta ahora

AUPR en el test fijo (89 fraudes; aleatorio = 0,0017). Las diferencias de ±0,05 no son concluyentes
con tan pocos fraudes: para elegir modelos se usa la validación cruzada.

| Modelo | AUPR CV (5 folds) | AUPR test | AUPR test sin duplicados | Fuente |
|---|---|---|---|---|
| GMM 5 componentes (partida del profesor) | 0,533 ± 0,077 | 0,567 / 0,495* | — / 0,463* | `docs/teoria/notebook_partida.md`, `02` §2, §12 |
| Regresión logística (`class_weight="balanced"`) | — | 0,685 | 0,658 | prueba de la celda común v2 |
| Mahalanobis (gaussiana única) | 0,655 ± 0,076 | 0,579 | 0,551 | `02` §3, §12 |
| GMM 5 `full` (RobustScaler), 22 variables | 0,734 ± 0,075 | 0,617 | 0,589 | `02` §4, §12 |
| GMM 5 · 1 eje menor (mejor no supervisado) | 0,779 ± 0,060 | 0,736 | 0,719 | `02` §9–§12 |
| Regresión logística (`C=0.1`, sin pesar) | 0,781 ± 0,073 | 0,703 | 0,677 | `03` §2, §9 |
| Random Forest 300 | 0,851 ± 0,060 | 0,793 | 0,776 | `03` §3, §9 |
| **XGBoost d3** (mejor supervisado) | **0,865 ± 0,060** | **0,814** | **0,801** | `03` §6–§9 |

\* El mismo GMM del profesor da 0,567 con su orden de filas y 0,495 con el orden por `row_id`: cambia la
inicialización por k-means (`02` §12).

Hallazgos clave del EDA (`01` §11): fraude 0,17 %, sin data shift entre construcción y producción,
fraudes como cola correlada en X17/X14/X12/X10 e importes exactos repetidos (1,50 = 24 % de los
fraudes).

Hallazgos clave del no supervisado (`02` §15): la señal vive en el **eje de varianza mínima de los
normales** (X17, X12, X14, X10 a la vez) y la representación importa más que el detector; ~1/6 de los
fraudes quedan dentro de la nube normal y ningún detector de rareza los ve. Hay un entregable de respaldo
en `data/output/02_scores_produccion_no_supervisado.csv` y cinco scores para el 04 en `02_scores.csv`.

Hallazgos clave del supervisado (`03` §12): **XGBoost d3 gana al mejor no supervisado** en los 5 folds y en
test (+0,078, IC 95 % 0,033–0,124). La logística solo lo iguala; lo que decide es la frontera no lineal.
**No se reequilibra**: pesos, SMOTE y undersampling no mejoran la AUPR. LightGBM por defecto diverge
(0,063) por `min_child_weight` = 0,001; configurado, empata con XGBoost. Recupera 15 de los 36 fraudes que
el 02 no veía (X17/X14 compensadas, X4 alta, importe bajo). Deja `03_scores.csv` (3 scores para el 04) y un
respaldo supervisado en `data/output/03_scores_produccion_supervisado.csv`.

**Siguiente paso:** `04_hibrido_y_ensemble` (batir 0,865 en CV con los scores del 02 + el XGBoost). Decisiones
abiertas en `docs/DECISIONES.md`: D-2.1 (enfoque final, se cierra en 04), D-3.1 (cabecera del CSV de entrega)
y D-3.3 (si el 99 rehace el preprocesado o lee el CSV del 01).

## Estado

- [x] Estructura del repositorio y datos
- [x] Documentación de teoría (`docs/teoria/`)
- [x] 01 EDA y preprocesado
- [x] 02 Baselines no supervisados
- [x] 03 Supervisados
- [ ] 04 Híbrido / ensemble y entregable
- [ ] 99 Entrega consolidada
