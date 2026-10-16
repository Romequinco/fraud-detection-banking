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
- **Estado:** ABIERTA

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
  supervisados). Distancias de Mahalanobis o modelos multivariantes, nunca euclídeas. El escalador solo
  afecta a euclídea y kNN (Mahalanobis y GMM-full no cambian con escalados lineales). Se ajusta dentro
  de cada fold en un `Pipeline` y se compara estándar frente a robusto en 02.
- **Estado:** CERRADA en la transformación; ABIERTA en el tipo de escalador (se decide en 02)

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
