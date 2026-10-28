# Registro de decisiones del proyecto

Formato ligero tipo ADR. ID `D-X.n`: `X` es el bloque (0 general, 1 datos/preprocesado,
2 modelado, 3 entregable) y `n` un correlativo. El estado es `ABIERTA` o `CERRADA`.

---

### D-0.1 · Partición de validación común

- **Fecha:** 2026-10-16
- **Decisión:** train / validación / test. **Test** = el 30 % del split estratificado del profesor
  (`random_state=42`, 89 fraudes), que solo se usa una vez, al final. **Validación** = rotatoria:
  `StratifiedGroupKFold(5)` sobre el 70 % de train (206 fraudes, ~41 por fold) agrupando duplicados
  (D-1.1). Se calcula una vez en 01 (columna `fold`) y 02–04 la leen con la celda común v2. Un val fijo
  se descartó porque con ~44 fraudes el AUPR oscila ±0,07. Está implementado
  en la celda común (no modificar) al inicio de cada notebook, sin código externo. Desde la v2,
  02–99 leen `data/output/01_dataset_preparado.csv`, que genera 01 (§12–16) y se entrega con el 99.
- **Justificación:** con 295 fraudes, el AUPR es muy ruidoso. Un split común hace comparables los
  resultados de todos los miembros.
- **Estado:** CERRADA

### D-2.1 · Enfoque de modelado: no supervisado, supervisado o híbrido

- **Fecha:** 2026-10-16
- **Alternativas:** (a) detección de anomalías entrenada solo con no-fraude (GMM de partida, IF, LOF,
  OCSVM, AE); (b) clasificador supervisado con las etiquetas; (c) híbrido, con los scores no
  supervisados como features del supervisado, y/o un ensemble.
- **Hipótesis inicial:** el supervisado o el híbrido deberían superar al no supervisado, porque hay
  etiquetas y el profesor recomienda usarlas cuando existen. Hay que verificarlo empíricamente en
  02–04.
- **Evidencia del 02 (2026-10-24):** el mejor no supervisado (D-2.4) da 0,779 ± 0,060 en CV y 0,736 en
  test, por encima de la regresión logística simple (0,685 en test). Deja fuera ~1/6 de los fraudes
  (dentro de la nube normal), que es lo que el 03/04 tienen que recuperar.
- **Evidencia del 03 (2026-10-26):** el mejor supervisado, XGBoost d3 (D-2.5), da 0,865 ± 0,060 en CV y
  0,814 en test: gana al GMM eje en los 5 folds y en test (+0,078, IC 95 % 0,033–0,124). La hipótesis inicial
  se confirma para el enfoque individual; queda por ver en el 04 si el híbrido añade algo.
- **Evidencia del 04 (2026-10-27):** comparación de tres entradas del mismo XGBoost d3, con scores
  recalculados mediante cinco folds externos y cinco internos por grupo: original 0,865, + GMM 0,864,
  + cinco scores 0,859 en CV. El híbrido no añade nada. Elección previa al test: **XGBoost original**
  (D-2.7).
- **Evidencia del 05 (2026-10-28):** ningún modelo único ni el ajuste de hiperparámetros mejoran al
  XGBoost; una mezcla por rangos de cuatro clasificadores supervisados sí, por poco (D-2.8). El enfoque
  final sigue siendo **supervisado**, sin scores no supervisados.
- **Estado:** CERRADA (04, D-2.7; modelo final en D-2.8)

### D-3.1 · Formato del CSV de producción

- **Fecha:** 2026-10-16
- **Decisión:** una columna con cabecera `score`, 113.923 filas en el orden del fichero
  de producción y sin índice. Si se lee con `pd.read_csv` sin `header=None`, la cabecera evita perder
  la primera fila.
- **Actualización (2026-10-28):** el 04 genera `scores_produccion.csv` con este formato, y también los
  respaldos del 02 y el 03. Se mantiene la cabecera.
- **Cambio final (2026-10-28):** el entregable va **sin cabecera**, al pie de la letra del enunciado ("un
  fichero csv con una sola columna (el score asignado a cada fila del fichero de producción)"): 113.923
  líneas, una por fila de producción, en su orden, sin índice. Lo genera el 99 con
  `save_submission()` (`header=False`). Los CSV de trabajo y de respaldo (`02_`, `03_`, `04_…`) conservan la
  cabecera `score`; no se entregan.
- **Estado:** CERRADA

### D-1.1 · Duplicados: agrupar, no eliminar

- **Fecha:** 2026-10-16
- **Evidencia (01 §3):** 4.167 filas duplicadas en construcción (10 entre fraudes), 0 conflictos de
  etiqueta; 2.109 en producción; 4.314 filas de producción idénticas a alguna de construcción.
- **Decisión:** se asigna un `group_id` a cada conjunto de filas idénticas y se hace la CV con
  `StratifiedGroupKFold` sobre `df_train`, para que las copias no crucen train y validación. El split
  fijo de la celda común no agrupa (1.643 filas de test, 5 de ellas fraudes, tienen copia en train): se
  marca `dup_en_train` y el AUPR de test se reporta con y sin esas filas. Producción se puntúa
  entera, en su orden y sin deduplicar.
- **Estado:** CERRADA (se implementa en 01 §12 y §15)

### D-1.2 · Transformación y escalado

- **Fecha:** 2026-10-16
- **Evidencia (01 §4, §6, §7):** colas muy pesadas (curtosis de Amount 466); `Amount` asimétrico y
  con un máximo mayor en producción (38.537 frente a 29.485). Fraudes fuertemente correlados entre
  sí (0,415 frente a 0,036).
- **Decisión:** `log1p(Amount)` y escalado ajustado solo con train (no-fraude en los métodos no
  supervisados). Distancias de Mahalanobis o modelos multivariantes, nunca euclídeas. El escalador se
  ajusta dentro de cada fold en un `Pipeline`.
- **Actualización (2026-10-24, 02 §4–§7):** el escalador **se elige por familia**. La frase original
  ("Mahalanobis y GMM-full no cambian con escalados lineales") solo es cierta para el modelo: el EM de un
  GMM `full` es invariante (con arranque aleatorio los tres escaladores dan 0,646), pero arranca de un
  k-means euclídeo que sí depende de la escala. Con RobustScaler el GMM 5 da 0,734 frente a 0,660 con
  StandardScaler (gana en 4–5 de 5 folds con 3 semillas). **Densidad (GMM): robust. Vecindad (kNN, LOF),
  OCSVM y PCA-reconstrucción: standard** (robust deja que `amount_freq` domine la distancia).
- **Estado:** CERRADA

### D-1.3 · Representatividad del test interno

- **Fecha:** 2026-10-16
- **Evidencia (01 §10):** KS construcción/producción ≤ 0,005 en todas las variables; AUC de la
  validación adversarial 0,496.
- **Decisión:** no hay data shift. El AUPR de test/CV se toma como estimación del de producción.
- **Estado:** CERRADA

### D-1.4 · Variables nuevas y dataset exportado

- **Fecha:** 2026-10-16
- **Evidencia (01 §6, §13):** importes exactos repetidos en fraude (1,50 = 24 % de los fraudes);
  mediana de `amount_freq` en train de 246 en fraudes frente a 79 en normales.
- **Decisión:** se añaden `log1p_Amount` y `amount_freq` (frecuencia del importe exacto en construcción
  + producción, **sin etiquetas**: transductivo pero sin fuga de la clase). Se exporta
  `data/output/01_dataset_preparado.csv` (284.807 × 29, versionado) con las variables sin estado,
  `split`, `fold`, `group_id` y `dup_en_train`; nada escalado. Conjunto por defecto en 02–04
  (`FEATURES` de la celda común v2): X1..X20 + `log1p_Amount` + `amount_freq`.
- **Estado:** CERRADA

### D-2.2 · Representación: cada detector con su conjunto de variables

- **Fecha:** 2026-10-24
- **Evidencia (02 §3, §6–§9):** la señal del fraude vive en la dirección de varianza mínima de los
  normales ("eje menor": X17, X12, X14, X10… a la vez). Con solo ese eje, IF pasa de 0,504 a 0,738,
  kNN de 0,533 a 0,746 y el GMM 5 de 0,734 a 0,779; PCA-reconstrucción con 21 de 22 componentes (= ese
  eje al cuadrado) da 0,745. ECOD sube de 0,583 a 0,748 con las top 4 + importe. El top-k por AUPR
  univariante es el mismo en el train de los 5 folds (sin fuga).
- **Decisión:** la representación importa más que el detector. Cada detector usa el conjunto elegido
  por CV en 02 §9; se prefieren los ejes menores (no usan etiquetas) salvo mejora clara. El eje menor es
  candidato a variable del 03/04.
- **Estado:** CERRADA

### D-2.3 · Detectores de vecindad: referencia submuestreada

- **Fecha:** 2026-10-24
- **Evidencia (02 §5):** LOF ajustado con todos los normales da 0,073; deduplicados, 0,123; con una
  submuestra de 20k normales, 0,693. Lo que decide es el tamaño del vecindario frente a la referencia
  (k/n), no tanto los duplicados.
- **Decisión:** kNN y LOF se ajustan con una submuestra de 20k normales del fold (k = 20, StandardScaler),
  sin deduplicar. Compatible con D-1.1: solo cambia el conjunto de ajuste del detector; lo que se
  puntúa nunca se deduplica.
- **Estado:** CERRADA

### D-2.4 · Mejor no supervisado y scores para el 04

- **Fecha:** 2026-10-24
- **Evidencia (02 §10–§12):** **GMM 5 · 1 eje menor** (RobustScaler + GMM 5 `full` sobre el eje menor):
  0,779 ± 0,060 en CV y 0,736 en test (IC 95 % 0,642–0,822), frente a 0,495 del GMM del profesor en el
  mismo test. Grupo de cabeza sin diferencias concluyentes: LOF 20 · 4 ejes menores (0,752) y ECOD · top 4
  + importe (0,748). Las combinaciones por rangos suman como mucho +0,013, casi lo mismo que un control
  sin diversidad (+0,008).
- **Decisión:** referencia no supervisada para el 03 = GMM 5 · 1 eje menor. Sin ensemble no supervisado.
  Pasan al 04 cinco scores poco correlados (ρ ≤ 0,46): GMM eje, LOF 4 ejes, ECOD top 4, GMM 5 full (22
  variables) y AE 12, en `data/output/02_scores.csv` (OOF en train; ajuste con los normales de train en
  test y producción).
- **Estado:** CERRADA

### D-3.2 · Entregable de respaldo

- **Fecha:** 2026-10-24
- **Decisión:** el 02 deja `data/output/02_scores_produccion_no_supervisado.csv` (GMM 5 · 1 eje menor
  reajustado con los normales de toda la construcción, mismo formato que D-3.1). Si el 03/04 no lo
  mejoran en CV, se entrega este. El entregable final (`scores_produccion.csv`) lo genera el 04.
- **Actualización (2026-10-28):** el modelo del entregable final lo fija el 05 (D-2.8), y el 99 lo
  regenera desde los CSV originales (D-3.3). Los respaldos son, en
  orden: `04_scores_produccion_xgb.csv` (XGBoost d3 del 04), `03_scores_produccion_supervisado.csv` y
  `02_scores_produccion_no_supervisado.csv`.
- **Estado:** CERRADA

### D-3.3 · ¿99 autocontenido o dependiente del CSV del 01?

- **Fecha:** 2026-10-24
- **Contradicción a resolver:** el README dice que el 99 "debe poder ejecutarse sin el resto del
  repositorio", pero la cabecera del 99 pide ejecutar antes el 01 y leer `01_dataset_preparado.csv`. Al
  profesor solo le llegan el `.ipynb` y el CSV de scores.
- **Alternativas:** (a) el 99 rehace el preprocesado del 01 (§12–§16) desde los dos CSV originales y luego
  usa la misma lógica que la celda común; (b) el 99 lee `01_dataset_preparado.csv` y se asume que el
  profesor tiene el repositorio.
- **Recomendación:** (a), porque el entregable solo incluye el notebook y el CSV de scores.
- **Actualización (2026-10-28):** el 04 cierra el modelo final, un XGBoost d3 sin scores del 02 (D-2.7).
  Con (a), el 99 no necesita reproducir los detectores del 02 para entrenarlo: le basta con el
  preprocesado del 01 (`log1p_Amount`, `amount_freq`, grupos de duplicados y folds) y el XGBoost. Es la
  única decisión abierta y lo primero que hay que resolver al montar el 99.
- **Decisión (2026-10-28):** **(a)**. El 99 lee solo `credit_card_construction.csv` y
  `credit_card_production.csv` (en la carpeta del notebook, en `data/` o en `../data/`), rehace el
  preprocesado del 01 y expone las mismas utilidades que la celda común v2. Si encuentra
  `01_dataset_preparado.csv`, comprueba que coincide (split, folds, grupos y variables): coincide. Entrena con
  los valores originales, no con el CSV redondeado del 01; el entregable resultante es prácticamente idéntico
  al del 05 (Spearman 1,000, diferencia máxima 8·10⁻⁵) y es el que se entrega.
- **Estado:** CERRADA

### D-2.5 · Mejor supervisado y scores para el 04

- **Fecha:** 2026-10-26
- **Evidencia (03 §2–§9):** logística 0,781–0,786 (empata con el GMM eje); Random Forest 0,851, Extra Trees
  0,841, MLP 0,790; **XGBoost d3 0,865 ± 0,060** en CV y 0,814 en test (IC 95 % 0,731–0,886). La AUPR es
  plana en profundidad (d = 2–6), ritmo de aprendizaje y semillas (± 0,002). El eje menor como variable no
  aporta a los árboles; quitar `amount_freq` cuesta 0,007.
- **Decisión:** referencia supervisada para el 04 = `XGBClassifier(n_estimators=400, learning_rate=0.05,
  max_depth=3, subsample=0.8, colsample_bytree=0.7, random_state=42)` sobre `FEATURES`. Pasan al 04 tres
  scores en `data/output/03_scores.csv` (XGBoost d3, RF, LR; OOF en train y ajuste con todo train en test y
  producción, mismo convenio que D-2.4). Respaldo supervisado en
  `data/output/03_scores_produccion_supervisado.csv` (XGBoost d3 reajustado con toda la construcción).
- **Estado:** CERRADA

### D-2.6 · Desbalanceo: no se reequilibra

- **Fecha:** 2026-10-26
- **Evidencia (03 §2, §4, §5):** sobre XGBoost, `scale_pos_weight` 10 empata (+0,001) y 50 baja (−0,003);
  SMOTE al 5 % baja (−0,009) y undersampling al 2 % baja más (−0,039, hasta −0,128 en un fold). En la LR,
  `class_weight="balanced"` también baja (−0,006). LightGBM por defecto diverge (0,063) por
  `min_child_weight` = 0,001: con ≥ 5 se recupera (0,862).
- **Decisión:** ningún reequilibrado (la AUPR solo mide el orden). En boosting, `min_child_weight` ≥ 1
  (el valor por defecto de XGBoost) es obligatorio.
- **Estado:** CERRADA


### D-2.7 · Comparación híbrida y modelo final

- **Fecha:** 2026-10-27
- **Alcance:** solo las tres comparaciones solicitadas: XGBoost original, XGBoost + score del GMM
  del eje menor y XGBoost + los cinco scores seleccionados en el 02. Se congela la configuración
  d3 del 03; no se prueban otros clasificadores, mezclas ni nuevos hiperparámetros.
- **Validación:** mismos cinco folds externos del 01. Dentro de cada entrenamiento externo se
  recalculan cinco folds internos por grupo para generar los scores OOF del entrenamiento híbrido;
  el fold externo se puntúa con detectores reajustados solo con los normales del entrenamiento externo.
  ECOD selecciona sus cuatro variables dentro del entrenamiento correspondiente. Los OOF globales del
  02 no se reutilizan como features de esa CV. Las configuraciones históricas del 02–03 permanecen
  seleccionadas previamente; el procedimiento no elimina ese optimismo.
- **Evidencia:** AUPR de CV: original **0,865 ± 0,061** (reproduce el 03), + GMM **0,864 ± 0,059**
  (Δ −0,0007, gana 4 de 5 folds) y + cinco scores **0,859 ± 0,060** (Δ −0,0056, gana 1 de 5). En test:
  **0,815**, **0,818** y **0,807**, respectivamente; los IC 95 % bootstrap de las diferencias frente al
  original incluyen el 0 (+0,0035 [−0,006, +0,015] y −0,0081 [−0,026, +0,007]). Cifras de
  `04_metricas_cv.csv` y `04_metricas_test.csv` (ejecución del 2026-10-27).
- **Prueba adicional (04 §5.1, 2026-10-28):** dar peso ×3 en `feature_weights` a la variable más predictiva o
  a las cuatro primeras (elegidas dentro de cada entrenamiento) da 0,864 y 0,865: Δ −0,0009 y −0,0000 frente
  al original. No se incorpora.
- **Criterio previo al test:** un híbrido debe mejorar la media al menos 0,005 y ganar en al menos
  cuatro de cinco folds. Se prefieren menos scores salvo mejora consistente de las cinco columnas
  frente a una. Es una regla práctica de estabilidad y mejora, no significación estadística.
- **Decisión:** **XGBoost original**. El test confirma y no cambia esa elección. Se reajusta con toda
  construcción (170.884 filas, 295 fraudes); si usa scores, se regeneran OOF sobre toda construcción
  y los detectores se reajustan con todas sus normales para puntuar producción.
- **Entregable:** `data/output/scores_produccion.csv`, 113.923 filas, una columna `score`, orden original.
  Configuración, versiones y firma del dataset en `04_seleccion_modelo.json`; métricas, scores y
  auditoría en `04_*.csv`. Las cifras reproducidas en 04 prevalecen para esta comparación sobre la
  narración histórica del 03. El test ya examinado en 02–03 no es una evaluación externa nueva.
- **Estado:** CERRADA (el modelo final del entregable lo sustituye D-2.8, en el 05)

### D-2.8 · Optimización final: mezcla por rangos de cuatro modelos

- **Fecha:** 2026-10-28
- **Alcance (05):** últimas mejoras sobre el XGBoost d3 de D-2.7: cinco semillas promediadas, más árboles con
  menos ritmo, búsqueda de hiperparámetros con Optuna (40 pruebas), CatBoost, LightGBM, Random Forest, tres
  variables nuevas sin etiquetas (`frec_fila`, `importe_entero`, `importe_99`) y cuatro mezclas por rangos con
  pesos iguales.
- **Validación:** 15 folds (la partición oficial del 01 y dos más con `StratifiedGroupKFold` por grupos), un
  placebo de semillas (el mismo XGBoost con semillas 43–46) para medir el ruido y una regla fijada antes de ver
  los candidatos: Δ medio ≥ +0,005, ganar en ≥ 11 de 15 folds y superar el mayor Δ medio de los placebos
  (0,0027). A Optuna, que busca sobre la partición oficial, se le exige además Δ ≥ +0,005 en las otras dos.
- **Evidencia:** referencia 0,858 ± 0,066 (el mismo XGBoost da 0,865 en el equipo del 04: no es idéntico entre
  equipos). Optuna 0,862 en la partición de búsqueda, pero −0,007 en las otras dos (sobreajuste). CatBoost
  0,861, LightGBM 0,862 y Random Forest 0,849, sin ganar de forma consistente. Variables nuevas, +0,002.
  Mezclas: XGB + CatBoost 0,864 (14 de 15), XGB + CatBoost + RF 0,865 y **XGB + CatBoost + LightGBM + RF 0,866 ±
  0,069 (Δ +0,008, 12 de 15)**. Robustez: +0,005 y 11 de 15 frente al XGBoost con 5 semillas. Test (después de
  elegir): 0,822 frente a 0,817 (Δ +0,005, IC 95 % −0,011 a +0,021); sin duplicados, 0,809.
- **Decisión:** modelo final = **media de rangos con pesos iguales** de XGBoost d3 (configuración de D-2.5),
  `CatBoostClassifier(iterations=1000, learning_rate=0.05, depth=6)`, `LGBMClassifier(min_child_weight=5)` y
  `RandomForestClassifier(300, min_samples_leaf=2)`, todos con semilla 42 y las 22 variables de `FEATURES`. Se
  reajustan con toda construcción y cada score se convierte en su rango dentro de producción.
- **Entregable:** `data/output/scores_produccion.csv` (113.923 × 1, sin cabecera desde el 99, D-3.1; valores en (0, 1]). El
  del 04 se guarda en `04_scores_produccion_xgb.csv`. Spearman 0,925 entre ambos; comparten el 63,9 % del 1 %
  superior. Las 6 copias exactas de fraudes de construcción en producción quedan en el 0,03 % superior.
- **Estado:** CERRADA
