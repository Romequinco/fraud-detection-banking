# Registro de decisiones del proyecto

Formato ligero tipo ADR. ID `D-X.n`: `X` es el bloque (0 general, 1 datos/preprocesado,
2 modelado, 3 entregable) y `n` un correlativo. El estado es `ABIERTA` o `CERRADA`.

---

### D-0.1 · Partición de validación común

- **Fecha:** 2026-10-16
- **Decisión:** split train/test 70/30 estratificado con `random_state=42` (el mismo del notebook de
  partida), más `StratifiedKFold(5)` con la misma semilla para seleccionar modelos. Está implementado
  en la celda común (no modificar) al inicio de cada notebook; sin código externo, para que el
  `99_ENTREGA` sea autocontenido.
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
