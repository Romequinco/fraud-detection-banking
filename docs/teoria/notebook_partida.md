# Notebook de partida, dataset y formato de entrega

> Fuentes: `docs/_fuentes/taller_deteccion_anomalias_partida.ipynb` (notebook del profesor, celdas
> 0–24), `docs/_fuentes/taller_anomalias.pdf` (enunciado) y los notebooks de clase del profesor
> (`scripts_classic/`, `scripts_deep/`) para el estilo de código. Las cifras marcadas como *(medido)*
> las ha calculado el grupo sobre `data/`.

## 1. Qué pide exactamente la entrega

| Requisito | Fuente | Cómo lo cumplimos |
|---|---|---|
| Práctica en grupos, entrega hasta las **18:30 del 29 de octubre** por el aula virtual | enunciado §Datos básicos | — |
| **Notebook `.ipynb`** con el **nombre de los componentes**, el **código desarrollado**, la **justificación de los desarrollos** y una **reflexión final sobre los resultados** | enunciado §4 | `notebooks/99_ENTREGA.ipynb`: cabecera con el grupo, secciones 1–5 con código y justificación, sección 6 de reflexión |
| **CSV con una sola columna**: el score asignado a cada fila del fichero de producción | enunciado §4 | `save_submission()` de la celda común → `data/output/scores_produccion.csv` |
| "Tantas filas como tiene el fichero de producción", y **las filas con mayor score son las de mayor probabilidad de fraude** | notebook de partida, celda 1 | 113.923 filas en el mismo orden que `credit_card_production.csv`. Se comprueba con un `assert` |
| Se mide el **AUPR en producción** (`sklearn.metrics.average_precision_score`) | notebook de partida, celda 1 | Toda la selección de modelos se hace por AUPR |
| Entorno Python | enunciado §5 | — |
| Evaluación: **resultados 75 %**, **notebook 25 %** | enunciado §6 | La prioridad es el AUPR; el notebook debe justificar cada decisión |

**Consecuencias prácticas:**

- `average_precision_score` solo depende del **orden** de los scores, no de su escala. Sirve
  cualquier score monótono: probabilidad, $-\log p(x)$, error de reconstrucción o rango.
- **Signo:** los modelos de densidad (`GaussianMixture.score_samples`) y los `score_samples` /
  `decision_function` de sklearn devuelven un valor **mayor = más normal**. Hay que **invertir el
  signo** antes de guardar: con el signo al revés, el AUPR se hunde por debajo del aleatorio.
- **Orden de filas:** no se puede reordenar, filtrar ni deduplicar producción antes de puntuar.
  Hay que puntuar las 113.923 filas en su orden original.
- **Cabecera del CSV:** pendiente (D-3.1 en `DECISIONES.md`). Ahora se guarda con la cabecera
  `score`.

## 2. El dataset

| | Construcción | Producción |
|---|---|---|
| Fichero | `data/credit_card_construction.csv` | `data/credit_card_production.csv` |
| Filas × columnas | 170.884 × 22 | 113.923 × 22 (`Class` vacía) |
| Features | `X1..X20` (anonimizadas), `Amount` | ídem |
| Fraudes | 295 (**0,1726 %**) | desconocido |
| Nulos | 0 | 0 en las features |

Observaciones de las salidas del notebook de partida (celdas 8 y 12):

- Las `X1..X20` están centradas en ~0. Su desviación típica decrece de forma monótona desde X1
  (390,9) hasta X20 (152,5), lo que apunta a componentes tipo PCA ordenadas por varianza (nota del
  redactor). Los valores parecen enteros pares escalados (p. ej. 374.0, -288.0).
- **`Amount`** está en otra escala (desviación típica 363,5, cola larga hasta miles) y no está
  centrada. Es candidata a log-transformación o escalado robusto (ver `teoria/01_metodos_clasicos.md`:
  Mahalanobis y escalas).
- Hay colas muy extremas: el mínimo de X2 es -14.544 y el de X8 es -14.644, con desviaciones típicas
  de ~300.

**Duplicados** *(medido)*: 4.167 filas completamente duplicadas en construcción (10 de ellas entre
los fraudes) y 2.109 en producción. En la celda 22 se ve el mismo score repetido para varios
fraudes (-64199,92 tres veces), lo que lo confirma. Tenerlo en cuenta en el 01 (EDA) y en la CV: un
duplicado puede caer a la vez en train y test.

## 3. Qué hace el notebook de partida (celda a celda)

| Celdas | Qué hace |
|---|---|
| 0–1 | Título y objetivo: maximizar el AUPR en producción y entregar el CSV de scores |
| 2, 5 | `COLAB = True`: descarga y descomprime los dos zips desde Google Drive con `wget` |
| 3 | Imports: pandas, numpy, tqdm, matplotlib, `roc_curve`, `precision_recall_curve`, `roc_auc_score`, `average_precision_score` |
| 6–9 | Carga de construcción y producción; `% Class` (99,83 / 0,17) y `describe()` |
| 10–11 | `plotly.express` scatter de una muestra de 10.000 filas, X1 vs X2, coloreado por `Class` (hover con `Amount`) |
| 12 | `df_const.std()` |
| 13–16 | `train_test_split(test_size=0.3, random_state=42, stratify=Class)`, comprobando que se conserva la proporción (0,1722 % en train, 0,1736 % en test) |
| 17–19 | `X_train/y_train/X_test/y_test`; `X_train0 = X_train[y_train==0]` ("datos ok"), con 119.412 filas |
| 20 | **Modelo estadístico de la distribución de los datos ok**: `GaussianMixture(n_components=5, random_state=42).fit(X_train0)` |
| 21 | `score_samples` = **logaritmo de la probabilidad** del punto |
| 22–23 | Ordena train y test por score: los fraudes aparecen arriba, con log-probabilidad muy negativa (hasta -65.102), y los normales hacia -95 |
| 24 | Vacía: aquí empieza el trabajo del grupo |

```python
# Núcleo del notebook de partida (celdas 13, 17, 18, 20, 22)
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture

df_train, df_test = train_test_split(df_const, test_size=0.3, random_state=42, stratify=df_const["Class"])
X_train = df_train.drop("Class", axis=1); y_train = df_train["Class"]
X_test = df_test.drop("Class", axis=1);   y_test = df_test["Class"]

X_train0 = X_train[y_train == 0]  # datos ok
gm_model = GaussianMixture(n_components=5, random_state=42)
gm_model.fit(X_train0)
scores_test = gm_model.score_samples(X_test)  # log p(x): MÁS BAJO = más anómalo
```

**Línea base medida** *(medido)*: el GMM de partida, con el signo invertido ($-\log p(x)$), da en el
test del split común **AUPR = 0,567** y **ROC-AUC = 0,948**. El AUPR aleatorio es la prevalencia,
**0,0017**. Este es el número que hay que batir.

Lectura del enfoque: es el enfoque "a menor probabilidad, más anómalo" de la clase
(`teoria/01_metodos_clasicos.md`), entrenado **solo con datos normales** (lo que la literatura suele llamar
*novelty detection* o detección semi-supervisada; nota del redactor). No usa las 295 etiquetas de fraude salvo para evaluar. El profesor dijo
en clase que, si se tienen etiquetas de anomalías, sirven sobre todo para **validar** umbrales y
modelos (ver `teoria/00_fundamentos.md` y `teoria/03_evaluacion_y_herramientas.md`).

## 4. Estilo de código del profesor (para que el 99 se le parezca)

Observado en el notebook de partida y en los `scripts_classic/` y `scripts_deep/`:

- Celdas cortas, una idea por celda, con comentarios breves en español (`# datos ok`,
  `# logaritmo de la probabilidad del punto`).
- Stack sencillo: `pandas`, `numpy`, `matplotlib`, `sklearn`. `plotly.express` para explorar.
  En deep, Keras/TensorFlow y PyTorch según el script. Usa la librería `adbench` para los datasets
  de ejemplo.
- Títulos markdown como secciones (`# Carga del dataset`). Inspecciona mostrando DataFrames
  directamente, sin `print`, y ordena por score para ver dónde caen las etiquetas.
- `random_state=42` en todo.
- Los notebooks `_clase` son las soluciones: el ejercicio sin resolver deja celdas vacías que la
  solución rellena.

## 5. Recomendaciones del profesor aplicables a la entrega

*(Lista consolidada a partir de los cuatro documentos de teoría; se completa al fusionar los
borradores.)*
