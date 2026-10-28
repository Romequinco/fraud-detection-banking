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
- **Cabecera del CSV:** **sin cabecera** (D-3.1 en `DECISIONES.md`): solo los 113.923 scores, uno por
  línea, como dice el enunciado.

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
**0,0017**. Este es el número que hay que batir. La mejor baseline no supervisada medida hasta ahora es un GMM de 10
componentes sobre `log1p(Amount)` + estandarizado, con **AUPR 0,662**. La tabla completa de distancias y
probabilidad está en `01_metodos_clasicos.md`, Parte A, "Aplicación a nuestra práctica". Con el dataset
preparado del 01 (`amount_freq` incluida) el GMM de 10 componentes llega a **0,696** y una regresión
logística simple a **0,685** (ver README, "Resultados hasta ahora").

**Actualización tras el 02 (2026-10-24):** el mejor no supervisado es un **GMM 5 sobre el eje de varianza
mínima de los normales**, con **0,779 ± 0,060 en CV** y **0,736 en test** (D-2.4). El mismo GMM de partida
da 0,567 con el orden de filas del profesor y 0,495 con el orden por `row_id`, porque cambia la
inicialización por k-means (`02` §12). Las cifras anteriores de este párrafo son de pruebas previas al 02.

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

Lista consolidada y sin repeticiones de lo que dijo el profesor en clase, ordenada según dónde se
aplica. Las citas literales y el resto de recomendaciones están en la sección "Lo que dijo el
profesor" de cada documento de `teoria/`.

**Planteamiento (justificarlo en el 99)**
- La definición de anomalía "depende de nosotros" y siempre falta algo que hay que fijar: umbral,
  distancia, distribución, modelo o vecindad [00:14:02], [02:51:08], [02:58:43]. → Escribir y
  justificar cada elección.
- "Casi cualquier clasificador os valdrá… con pequeñas modificaciones"; "con lo que sabéis podéis
  resolverlo sin nada nuevo" [01:01:52], [02:51:23]. → No hace falta exotismo: lo que se premia es
  el AUPR.
- "En realidad todo es lo mismo" (distancia, probabilidad, información, modelo): cada enfoque falla
  si la elección de base no encaja con los datos [02:50:53]. → Comparar varias familias.

**Datos (01 EDA)**
- "Si algo se puede visualizar, se visualiza": entender primero en qué caso estamos [03:42:14].
- Pintar el histograma antes de asumir una gaussiana [01:46:15]. Transformar una variable, por
  ejemplo con un logaritmo, cambia qué es outlier [00:12:51]. → `Amount`.
- Mahalanobis frente a euclídea: con escalas distintas, la euclídea "ordena mal los puntos";
  normalizar primero equivale a usar Mahalanobis [01:29:39], [01:30:19].
- Duplicados y features irrelevantes pueden romper métodos (de distancia y de probabilidad) [03:48:01].
- Si conocéis propiedades del dominio, "metedlas en el modelo" [00:57:33].

**Etiquetas y validación (02–04)**
- **Con muy pocas anomalías etiquetadas, usarlas para validar más que para entrenar**; solo con
  tantas como normales tiene sentido entrenar con ellas [01:51:46], [02:45:03]–[02:45:35].
  → Nosotros tenemos 295 fraudes: comparar empíricamente los dos usos (D-2.1) y justificar la elección.
- Con pocas etiquetas, supervisado ≈ no supervisado; lo que más cambia el resultado es **el tipo de
  anomalía** (del 50 % al 100 % según el método) [03:46:00]–[03:47:11]. No free lunch [03:47:24].
- Desbalanceo: undersampling (tira información), oversampling o SMOTE ("un poco más inteligente"),
  asumiendo que los outliers vienen de "un millón de clases que nunca habéis visto" [02:48:55]–[02:50:04].
- Escepticismo con los resultados: más reciente no significa mejor [03:45:07]; mirar los ejes y no
  fiarse de las negritas [03:40:58], [03:46:27].

**Modelos**
- Probabilidad: con un buen modelo de densidad, mejor que distancias [01:39:33]. Nunca buscar
  anomalías cerca de los picos [01:41:20]. El 95/99 % es una convención humana [01:49:45].
- "El modelo tiene que ser suficientemente complicado" y hay que saber si es bueno [02:41:01].
- Isolation Forest está "diseñado justo para anomaly detection" [03:00:36]. Los métodos de
  vecindad (LOF, DBSCAN) "escalan muy mal" [02:59:58].
- Autoencoder: error de reconstrucción o espacio latente + modelo clásico sencillo, que "funciona
  muy bien" y es interpretable [03:52:25]–[03:54:11].
- Deep y generativos: las mejores soluciones si hay cómputo y datos, pero más caros y obligan a
  tomar más decisiones [03:58:28], [04:13:55].

**Explicación y producción (99, reflexión final)**
- Si se puede, explicar por qué algo es outlier: umbral y variables que causan la distancia
  [03:24:27]. No usar LIME, SHAP ni contrafactuales directos sobre outliers [03:33:16].
- Data shift: "va a pasar mucho en la práctica", también en finanzas. Guardar validación en
  producción y vigilar [03:10:18], [03:14:04]. → Comprobar construcción frente a producción en el 01
  y comentarlo en la reflexión.
- "Siempre se suele acertar mirando para atrás, pero nunca mirando para delante" [01:00:29].
  → Ser prudente con lo que el AUPR de test promete para producción.
