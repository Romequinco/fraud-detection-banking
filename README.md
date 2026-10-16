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

```bash
pip install -r requirements.txt
jupyter lab notebooks/
```

- **01** genera `data/output/01_dataset_preparado.csv`: split fijo train/test (el del profesor),
  `fold` de validación (StratifiedGroupKFold por duplicados) y variables sin estado.
- **02–04 y 99** empiezan con la misma **celda común v2 (no modificar)**: leen ese CSV y exponen
  `X_train/X_test/X_prod`, `cv_splits`, `make_pipe()` (escalado dentro del fold), `cv_aupr()`,
  `report()` (AUPR de test con y sin duplicados), `plot_pr()` y `save_submission()`. Así los AUPR de
  todos los miembros son comparables.
- Cada notebook guarda sus tablas en `data/output/NN_*.csv`.
- Las decisiones relevantes se registran en `docs/DECISIONES.md`.
- `99_ENTREGA.ipynb` consolida al final el código real de `01`–`04` y debe poder ejecutarse sin el
  resto del repositorio.

## Resultados hasta ahora

AUPR en el test fijo (89 fraudes; aleatorio = 0,0017). Las diferencias de ±0,05 no son concluyentes
con tan pocos fraudes: para elegir modelos se usa la validación cruzada.

| Modelo | AUPR test | AUPR test sin duplicados | Fuente |
|---|---|---|---|
| GMM 5 componentes (partida del profesor) | 0,567 | — | `docs/teoria/notebook_partida.md` |
| Regresión logística (`class_weight="balanced"`) | 0,685 | 0,658 | prueba de la celda común v2 |
| GMM 10 componentes + `log1p_Amount` + `amount_freq` | 0,696 | 0,670 | prueba de la celda común v2 |

Hallazgos clave del EDA (`01` §11): fraude 0,17 %, sin data shift entre construcción y producción,
fraudes como cola correlada en X17/X14/X12/X10 e importes exactos repetidos (1,50 = 24 % de los
fraudes).

**Siguiente paso:** `02_baselines_no_supervisados`. Decisiones abiertas en `docs/DECISIONES.md`:
D-2.1 (enfoque final) y D-3.1 (cabecera del CSV de entrega).

## Estado

- [x] Estructura del repositorio y datos
- [x] Documentación de teoría (`docs/teoria/`)
- [x] 01 EDA y preprocesado
- [ ] 02 Baselines no supervisados
- [ ] 03 Supervisados
- [ ] 04 Híbrido / ensemble y entregable
- [ ] 99 Entrega consolidada
