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
- **Estado:** ABIERTA (se cierra en 04)

### D-3.1 · Formato del CSV de producción

- **Fecha:** 2026-10-16
- **Decisión provisional:** una columna con cabecera `score`, 113.923 filas en el orden del fichero
  de producción y sin índice.
- **Pendiente:** confirmar si se quiere la cabecera. Si se lee con `pd.read_csv` sin `header=None`,
  la cabecera evita perder la primera fila.
- **Estado:** ABIERTA

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
- **Estado:** ABIERTA (decidir antes de montar el 99)

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
