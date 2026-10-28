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
    ├── 05_optimizacion_final.ipynb
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
- **02–05** empiezan con la misma **celda común v2 (no modificar)**: leen ese CSV y exponen
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
- **04** reproduce el XGBoost del 03, prueba los híbridos con scores recalculados dentro de cada fold
  (hay una caché local, regenerable) y deja `04_metricas_cv.csv`, `04_aupr_por_fold.csv`,
  `04_metricas_test.csv`, `04_intervalos_test.csv`, `04_captura_oof.csv`, `04_importancia.csv`,
  `04_scores.csv`, `04_auditoria_scores.csv`, `04_prioridad_*.csv` (prueba de prioridad de variables, §5.1),
  `04_seleccion_modelo.json` (configuración y versiones) y el
  entregable del XGBoost (ahora respaldo en `04_scores_produccion_xgb.csv`).
- **05** tarda unos 16 min (búsqueda con Optuna, CatBoost y Random Forest en 15 folds). Necesita además
  `catboost` y `optuna`. Deja `05_metricas_cv.csv`, `05_aupr_por_fold.csv`, `05_metricas_test.csv`,
  `05_optuna.csv`, `05_seleccion_modelo.json` y el **entregable final `scores_produccion.csv`**.
- Las decisiones relevantes se registran en `docs/DECISIONES.md`.
- **99** (`99_ENTREGA.ipynb`, el notebook que se entrega) consolida `01`–`05` en 160 celdas y se ejecuta
  de principio a fin **solo con los dos CSV originales** (unos 17 min): rehace el preprocesado del 01 (y, si
  encuentra `01_dataset_preparado.csv`, comprueba que coincide), re-ejecuta los experimentos que justifican
  cada decisión, resume en tablas todo lo demás que se probó y genera `scores_produccion.csv`. Busca los CSV
  en la carpeta del notebook, en `data/` o en `../data/`. Necesita `xgboost`, `lightgbm` y `catboost` (no
  `pyod`, `imbalanced-learn`, `optuna` ni `torch`).

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
| XGBoost d3 (mejor supervisado) | 0,865 ± 0,060 | 0,814 | 0,801 | `03` §6–§9 |
| XGBoost d3 + score GMM eje (híbrido) | 0,864 ± 0,059 | 0,818 | 0,805 | `04` §4, §9 |
| XGBoost d3 + cinco scores del 02 (híbrido) | 0,859 ± 0,060 | 0,807 | 0,793 | `04` §5, §9 |
| XGBoost d3 reproducido en el 04 | 0,865 ± 0,061 | 0,815 | 0,801 | `04` §1, §8–§9 |
| XGBoost d3 en el 05 (15 folds, otro equipo) | 0,858 ± 0,066 | 0,817 | 0,804 | `05` §1, §9 |
| XGBoost ajustado con Optuna | 0,855 ± 0,067 | 0,820 | 0,808 | `05` §3, §9 |
| **Mezcla por rangos XGBoost + CatBoost + LightGBM + RF (modelo final)** | **0,866 ± 0,069** | **0,822** | **0,809** | `05` §6–§9 |

\* El mismo GMM del profesor da 0,567 con su orden de filas y 0,495 con el orden por `row_id`: cambia la
inicialización por k-means (`02` §12). Las filas del `05` usan 15 folds (3 particiones × 5) en lugar de 5.

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

Hallazgos clave del híbrido (`04` §12): **los scores del 02 no aportan nada al XGBoost**. Con los scores
recalculados dentro de cada fold (5 externos × 5 internos por grupo, sin fuga), añadir el GMM eje da −0,0007
en CV (gana 4 de 5 folds) y añadir los cinco scores da −0,0056 (gana 1). Ninguno cumple la regla fijada antes
del test (Δ ≥ 0,005 y ≥ 4 de 5 folds), así que el modelo final es el **XGBoost d3 original** (D-2.7). Dar
más probabilidad de muestreo a las variables más predictivas (`feature_weights` ×3, §5.1) tampoco mejora. En test
las diferencias tienen IC 95 % que incluyen el 0. El XGBoost se reajusta con toda la construcción (170.884
filas, 295 fraudes) y genera el entregable del XGBoost.

Hallazgos clave de la optimización final (`05` §12): con 15 folds, un placebo de semillas y una regla fijada
de antemano (Δ ≥ +0,005, ≥ 11 de 15 folds y por encima del ruido de semilla), **ajustar hiperparámetros no
mejora** (Optuna gana en los folds donde busca y pierde en los demás), **ningún modelo único bate al XGBoost**
y las variables nuevas no aportan. Lo que sí mejora, poco pero de forma consistente, es **mezclar modelos
distintos**: la media de rangos con pesos iguales de XGBoost d3, CatBoost, LightGBM y Random Forest da 0,866
en CV (Δ +0,008, gana en 12 de 15 folds; +0,005 frente al XGBoost con 5 semillas) y 0,822 en test (Δ +0,005,
IC −0,011 a +0,021). Es el **modelo final** (D-2.8): `data/output/scores_produccion.csv` es ahora la mezcla
reajustada con toda construcción, y el XGBoost del 04 queda como respaldo en `04_scores_produccion_xgb.csv`.
El mismo XGBoost no da resultados idénticos en equipos distintos (0,865 en el del 04, 0,858 aquí), otra razón
para preferir la mezcla, que es más estable.

Entrega consolidada (`99`): reproduce desde los CSV originales todas las cifras clave (preparación idéntica a
la del 01; GMM del profesor 0,533; GMM eje menor 0,779 / 0,736; XGBoost 0,858 / 0,817; mezcla 0,866 / 0,822) y
genera un `scores_produccion.csv` prácticamente idéntico al del 05 (Spearman 1,000; diferencia máxima 2·10⁻⁵).
En este equipo, el híbrido XGBoost + score del GMM eje da Δ +0,005 en los 5 folds oficiales, pero gana solo en
3 de 5 (8 de 15 en la validación ampliada): tampoco cumple la regla, como en el 04.

**Para entregar (29-oct, 18:30):** `notebooks/99_ENTREGA.ipynb` y `data/output/scores_produccion.csv`.

## Estado

- [x] Estructura del repositorio y datos
- [x] Documentación de teoría (`docs/teoria/`)
- [x] 01 EDA y preprocesado
- [x] 02 Baselines no supervisados
- [x] 03 Supervisados
- [x] 04 Híbrido / ensemble y entregable (`scores_produccion.csv` generado)
- [x] 05 Optimización final (mezcla de cuatro modelos; nuevo `scores_produccion.csv`)
- [x] 99 Entrega consolidada (autocontenida, ejecutada; genera el entregable)
