# Evaluación, comparativa de métodos (ADBench), OOD, explicabilidad de outliers y herramientas

> **Fuentes**
> - Diapositivas `slides_clasicas/p49.png`–`p57.png` (sección "Comparativa": ADBench, Han et al., NeurIPS 2022). Revisadas una a una sobre la imagen.
> - `transcripcion.txt` líneas 1940–2219 ([03:24:03]–[03:50:01]) y 2522–2601 ([04:21:03]–[04:30:47]). Además, tramos de evaluación que estaban en otras partes de la clase: [01:51:28]–[01:51:57], [02:45:03]–[02:52:06], [03:09:27]–[03:18:28] y [00:43:42]–[00:44:21].
> - `papers/45_Counterfactual_generation_f.txt`: Keshtmand, Santos-Rodriguez y Lawry, *Counterfactual generation for Out-of-Distribution data* (Univ. Bristol; según el borrador, NLDL 2026 / PMLR 307; la cabecera del txt no indica la sede).
> - `papers/keshtmand24a.txt`: Keshtmand, Santos-Rodriguez y Lawry, *Typicality-based point OOD detection with contrastive learning* (NLDL 2024, PMLR 233).
> - `papers/1907.00811v1.txt`: Wang, Mavromatis, Tassi, Santos-Rodríguez y Piechocki, *Location Anomalies Detection for Connected and Autonomous Vehicles* (arXiv 2019).
>
> - Sección 23 (herramientas): notebooks `AD_Tools_Sklearn(_clase)` y `AD_Tools_Pyod` (API verificada ejecutando sklearn 1.8; PyOD según documentación y salidas del notebook).
>
> **Convenciones.** "PROF" = Raúl Santos-Rodríguez (Bristol, coautor de los tres papers). La transcripción automática etiqueta todo como PROF, también las preguntas de los alumnos. He limpiado las citas de errores fonéticos: "Country factions", "capefactos", "cante factor", "calefacto", "artefactos" y "Camper Patchwols" = *counterfactuals*; "online", "line" y "color lines" = LIME; "share" = SHAP; "auto distribution", "auros distribution" e "indistitution" = *out-of-distribution* / *in-distribution*; "aulayer", "outlife", "Outl Lion", "ondayas" y "outlives" = *outlier(s)*; "Andy", "andes" y "embist" = MNIST; "Kay andes" = KMNIST; "nor freelance" = *no free lunch*; "datos iluminados" = datos duplicados; "Dakes of Shift" y "chip o de ir a selship" = *dataset shift* / *covariate shift*; "Smoke" = SMOTE; "clases bien" = One-Class SVM; "bajos Words" = *bag of words*; "Majadanovich" = Mahalanobis. Lo que no viene de las fuentes va marcado **(nota del redactor)**.

---

## 1. Explicabilidad de outliers: por qué hay que explicar

**Idea.** No basta con que el detector diga "este dato es anómalo": conviene explicar **por qué**. El profesor lo presenta como algo "que está muy bien hacer, si se puede, en detección de outliers". No vale con decir "este dato me ha parecido aquí, no sé qué es": hay que intentar explicarlo.

**Detalle.** La explicación más sencilla es el **propio umbral**: "tengo un umbral que son 5 y esto estaba a una distancia de 6". A partir de ahí se busca "qué características dentro de ese vector, dentro de esa muestra, son las que han hecho que esta distancia sea esa". Es decir, una atribución por feature de la puntuación de anomalía.

- (nota del redactor) Formalmente: si el score es una distancia $s(\mathbf{x})$ y la regla es $s(\mathbf{x}) > \tau$, la explicación mínima es el margen $s(\mathbf{x})-\tau$ más la descomposición de $s$ por variables. En Mahalanobis, $(\mathbf{x}-\boldsymbol\mu)^\top\Sigma^{-1}(\mathbf{x}-\boldsymbol\mu)=\sum_d l_d^2/\lambda_d$, con $l_d=\nu_d^\top(\mathbf{x}-\boldsymbol\mu)$, se reparte por componentes principales; es la misma descomposición que usa el paper de tipicalidad (paper B). En un GMM se puede usar la componente más responsable y los z-scores por feature respecto a ella.

*Fuente: transcripción [03:24:27]–[03:24:55]*

---

## 2. Contrafactuales clásicos (recordatorio)

**Idea.** Un contrafactual es el punto más cercano **al otro lado de la frontera de decisión**, es decir, el cambio mínimo en la entrada que cambia la predicción. Se valoran porque son **accionables**.

**Fórmulas (paper OOD CF, §2.1).** $\mathbf{x}'=\mathbf{x}+\boldsymbol\delta$, donde la perturbación $\boldsymbol\delta$ cruza la frontera. El método clásico es CFI (Wachter et al., 2017):

$$L(\mathbf{x}',\mathbf{x})=\big(\hat P_t(\mathbf{x}')-P_t\big)^2+\lambda\,\lVert\mathbf{x}'-\mathbf{x}\rVert_1$$

$\hat P_t(\mathbf{x}')$ es la probabilidad predicha de la clase objetivo $t$ y $P_t$ la probabilidad deseada (normalmente 1). El término $L_1$ favorece la *sparsity*.

**Propiedades deseables** (§2.1): realismo, *sparsity* (pocas features cambiadas; *block sparsity* = las features relacionadas cambian juntas), factibilidad, proximidad y cercanía a la variedad de los datos.
**Variantes:** **CEM** (Dhurandhar et al.) añade un regularizador *elastic net* $L_1+L_2$ y el error de reconstrucción de un autoencoder. **Proto** (Van Looveren y Klaise) hace lo mismo y añade un término que acerca el contrafactual, en el espacio latente, al **prototipo** de la clase objetivo.

**Ejemplos del profesor.** Dos clases (círculos y triángulos) y un outlier. Para un punto normal, el contrafactual es "el triángulo más cercano al otro lado de la frontera". El ejemplo del banco: pido un préstamo, me lo deniegan y quiero saber "cuál es la persona a la que se la han dado que es más parecida a mí". "Una persona como tú, pero que gana 1.000 € más al mes": el cambio es **accionable**.

*Fuente: transcripción [03:25:14]–[03:26:54] · paper OOD CF pp. 2–3*

---

## 3. Por qué los contrafactuales directos fallan con outliers

**Idea.** Si el punto es un **outlier lejano** a todas las clases, el contrafactual tradicional es un punto "que no existe" en los datos y que no se relaciona con ninguna clase. Es decir, "estoy intentando explicar un outlier con otro outlier que me acabo de inventar", y "tampoco sé interpretar" la explicación.

**Paper (Fig. 1, ejemplo de juguete).** Los datos ID son triángulos y círculos azules, y la frontera separa solo por **forma**. Un **hexágono rosa** es far-OOD: difiere en forma (feature **discriminativa**) y en color (feature **no discriminativa**). Wachter solo cambia la forma y produce un "óvalo rosa", que cruza la frontera **pero sigue siendo OOD** por el color.

*Fuente: transcripción [03:27:07]–[03:27:46] · paper OOD CF pp. 1–3, Fig. 1*

---

## 4. Contrafactuales en dos etapas para outliers (OOD counterfactuals)

**Idea.** (1) Primero se **acerca el outlier a una zona con densidad de datos** ("algún sitio que podamos entender un poco más") y se explica cómo se transforma. (2) Una vez allí, se busca el contrafactual de la clase contraria "que tenga algo de sentido con respecto a nuestros datos". Es "una explicación en 2 etapas, que es bastante inusual, pero es bastante útil cuando os encontréis outliers y queráis explicarlos". El profesor lo conecta con los métodos de distancia: "voy a acercarme a cualquiera de las clases y ahora mi dato se vuelve más probable; y una vez que esté allí, voy a buscar la clase contraria".

**Método (paper OOD CF, §3–4).**
- Se distinguen tres tipos de dato. **Far-OOD** difiere en features discriminativas *y* no discriminativas. **Near-OOD** comparte las no discriminativas, pero no las discriminativas. **ID** es el dato en distribución. La trayectoria es far-OOD → (optimizar $z_n$) → near-OOD → (optimizar $z_d$) → ID (Fig. 2).
- El objetivo ideal es $D_{ID}(\phi_z(\mathbf{x}+\boldsymbol\delta))<D_{OOD}(\phi_z(\mathbf{x}+\boldsymbol\delta))$ (Ec. 3). Como no hay datos OOD en entrenamiento, se usa $D_{ID}(\phi_z(\mathbf{x}+\boldsymbol\delta))<D_{ID}(\phi_z(\mathbf{x}))$ (Ec. 4), con $\mathbf{z}=(\mathbf{z}_n;\mathbf{z}_d)$ y $\boldsymbol\delta=\boldsymbol\delta_n+\boldsymbol\delta_d$ (Ec. 5–6).
- Pasos:
  1. Extraer los latentes con **PCA** (vale cualquier extractor).
  2. Partir $\mathbf{z}$ en $\mathbf{z}_d$ y $\mathbf{z}_n$ con **selección secuencial voraz** (`SequentialFeatureSelector` + **QDA**). Se toma $k$ = la mitad de las features como discriminativas.
  3. Descender por gradiente la **NLL bajo un GMM**. Para $\mathbf{z}_n$ se usa una sola componente; para $\mathbf{z}_d$, tantas componentes como clases ID.
- Pérdidas: $L_{CF,n}=-\log p_n(\phi_n(\mathbf{x}))$ y $L_{CF,d}=-\log p_d(\phi_d(\mathbf{x}))$. Actualización: $\mathbf{x}'=\mathbf{x}-\alpha\nabla_{\mathbf{x}}L$, con $\alpha$ elegido por validación cruzada (Ec. 7–10). Opcionalmente se añaden $L_1$ y un término de reconstrucción con autoencoder (Ap. B).
- ¿Por qué un GMM y no flows o EBMs? (Ap. A) Porque la optimización es estable: los gradientes apuntan a la gaussiana de la clase. Los flows y los EBMs pueden asignar densidad alta lejos de los datos.
- Los autores no lo presentan como solución completa: por defecto no incorpora ni *sparsity* ni factibilidad.

**Ejemplos del profesor.** Perros contra gatos más un **caballo**: primero se explica cómo el caballo se mueve hacia el perro o el gato; después viene el contrafactual. Reformulado con **lobos contra leones**: "nos acercamos primero a la clase de león… porque derivan de un ancestro común; luego, con el contrafactual, cojo uno de la clase lobo para decir: esto no es un lobo por esta razón". La idea general: "idealmente encuentras algo en común que te permite acercarte".

*Fuente: transcripción [03:27:46]–[03:30:24] · paper OOD CF pp. 3–5, Fig. 2–3, Alg. C.1, Ap. A–B*

---

## 5. Por qué LIME y SHAP fallan con outliers

**Repaso de LIME (profesor).** Se tiene acceso a la **caja negra** y a un punto. Se **muestrea alrededor del punto añadiendo ruido**, cada vez más lejos, y se etiquetan esas muestras con la predicción del modelo. Con ellas se ajusta "un modelo muy simple, lineal o un árbol de decisión", que aproxima la frontera **localmente**. Se muestrea con ruido "porque no tengo acceso a los datos, simplemente a la caja negra".

**Problemas con outliers.** (1) El outlier "no está nada cerca de la frontera, con lo cual la aproximación a la frontera va a ser mala". (2) Al muestrear alrededor, "ni siquiera cogemos ningún punto de la otra clase. Y aunque lo cogiésemos, sería un muestreo tan grande que tendría una aproximación muy mala a la frontera". Conclusión: primero se lleva el punto a la clase y se entiende el camino, y "luego ya puedo utilizar LIME o SHAP o lo que queráis".

*Fuente: transcripción [03:30:31]–[03:34:17]*

---

## 6. AI Act europeo y el ejemplo hipoteca/edad

**Idea.** Si la regulación exige explicaciones, un sistema que no gestiona outliers puede "cumplir" con contrafactuales válidos pero **irrelevantes**: "siempre voy a encontrar [uno]. Solo tengo que ir al punto más cercano de la otra clase en línea recta… y decir que es mi contrafactual, pero es completamente irrelevante. Habré justificado que tengo una explicación."

**Ejemplo.** "El sueldo tiene que ser 10.000 € mayor y tener 20 años menos." "No tiene ningún sentido, porque yo no puedo tener 20 años menos": esa persona debería haberse tratado como outlier ("usted no debería pedir una hipoteca").

**Dos salidas correctas:** (a) detectar el outlier y **no explicar** ("no te voy a dar una explicación porque es un outlier y esto no va a funcionar con este modelo"); (b) detectarlo y **explicarlo de otra manera**, en dos etapas. "Puede fallar por las 2 cosas, o bien no detectar o bien no explicar bien."

*Fuente: transcripción [03:34:20]–[03:36:06] · paper OOD CF p. 1*

---

## 7. Evaluar con datos sintéticos: MNIST frente a KMNIST, FashionMNIST y ColoredMNIST

**Idea.** En investigación OOD es habitual tomar un dataset como **in-distribution** y otro como **OOD**. Así "tienes etiquetas reales de lo que es fuera de distribución, pero no es un problema muy real": "vale muy bien para evaluar, pero no vale demasiado para el mundo práctico". Con ironía: "podemos distinguir caracteres japoneses de números, que es una cosa muy avanzada". Además, "en el mundo de los contrafactuales, los experimentos son siempre horribles".

**Detalle.** KMNIST (Kuzushiji-MNIST, caracteres japoneses manuscritos; 28×28 en gris, 70.000 imágenes), FashionMNIST (prendas; solo lo menciona el profesor), ColoredMNISTRed (canales azul y verde a cero). El paper genera la primera etapa (acercar a la distribución) y el contrafactual "real" en la segunda (Fig. 5). En el paper de tipicalidad, los ID son CIFAR100, Caltech256 y TinyImageNet, y los OOD son esos mismos más CIFAR10, SVHN y MNIST (más FashionMNIST y KMNIST en el apéndice).

*Fuente: transcripción [03:36:11]–[03:38:00] · paper OOD CF pp. 6–7 · keshtmand24a p. 4 y Ap. A.4*

---

## 8. "Si algo se puede visualizar, se visualiza": entender en qué caso estamos

**Idea.** "Acordaos que a la hora de evaluar, como vimos ayer, vamos a tener muchas métricas, pero algunas serán más relevantes que otras, dependiendo de qué tipo de problema tengáis. Entonces el consejo siempre es: si algo se puede visualizar, se visualiza. Intentamos entender en qué caso estamos." Ejemplo: el contrafactual en dos etapas "no tiene nada que ver si lo que tengo son *local outliers*" que ya están cerca de la clase; ahí sirve el método estándar. "Siempre como primer paso, intentar entender en qué situación estamos." El mismo consejo aparece al cerrar el taller: "siempre eso de visualizar" [04:19:38].

**Escepticismo.** Al leer papers, "los números en negrita corresponden a los métodos que estamos proponiendo… y ya todo el mundo contento". Con las gráficas, "los ejes empiezan y acaban donde cada uno le da la gana". Sobre ADBench: "todos ellos funcionan mejor que aleatorio" [03:46:31].

- La frase transcrita es "a la hora de evaluar es partido de ayer". La interpreto como "como vimos ayer": las **métricas se explicaron en la sesión anterior**, que no está en esta transcripción.

*Fuente: transcripción [03:41:51]–[03:42:41], [03:40:58]–[03:41:10], [03:46:18]–[03:46:31]*

---

## 9. Timeline del campo (slide p51)

**Idea (profesor).** Primero, aproximaciones **estadísticas** ("la campana de Gauss y desviaciones típicas"). Después, **métodos de distancia**. Después, "cualquier clasificador": árboles de decisión ("es fácil separar un punto") y SVM ("normalmente será el One-Class SVM"). Todo eso es anterior a 2017; luego llegan las redes, y "desde 2022 casi todo lo llamamos *out-of-distribution* en vez de *outlier detection*". "No porque sea más reciente funciona mejor."

**Slide p51 (30 algoritmos):**
- **14 no supervisados.** Antes de 2017: COF (Tang), IForest (Liu), LODA (Tomáš Pevný), KNN (Ramaswamy), PCA (Shyu), HBOS (Goldstein), LOF (Breunig), CBLOF (He), SOD (Kriegel) y OCSVM (Schölkopf). En 2018: DAGMM (Zong) y DeepSVDD (Ruff). Hacia 2020–21: COPOD (Li). En 2022: ECOD (Li).
- **7 semi-supervisados:** XGBOD (Zhao) y REPEN (Pang) hacia 2018; GANomaly (Akcay) entre 2018 y 2019; DevNet (Pang) entre 2019 y 2020; PReNet (Pang) y DeepSAD (Ruff) hacia 2020; FEAWAD (Zhou) en 2021.
- **9 supervisados.** Antes de 2017: Naive Bayes (Bayes), SVM (Vapnik), MLP (Rosenblatt) y RF (Leo Breiman). Hacia 2017: XGB (Chen) y LGB (Ke). Hacia 2019: CatB (Prokhorenkova). Hacia 2021: ResNet y FTTransformer (Gorishniy).
- Los años son aproximados: se leen de la posición de las líneas sobre el eje.

El profesor anuncia un ejercicio ("a ver si sois capaces de hacer lo mismo que [un paper de] 2019 en 5 minutos en clase") que pertenece al bloque siguiente, de deep learning.

*Fuente: slide p51 · transcripción [03:42:53]–[03:44:04], [03:45:02]–[03:45:20]*

---

## 10. ADBench: diseño y tipos de anomalía (slides p49, p50, p52)

**Diseño (p52).** 57 datasets de benchmark y 30 algoritmos (ambos en la secc. 3.2 del paper), analizados desde tres ángulos:
- **Ángulo I, nivel de supervisión** (3.3.1 y 4.2): no supervisado, semi-supervisado y totalmente supervisado.
- **Ángulo II, tipos de anomalía** (3.3.2 y 4.3): local, global, dependencia y clustered.
- **Ángulo III, datos ruidosos y corruptos** (3.3.3 y 4.4): anomalías duplicadas, features irrelevantes y errores de anotación.

La slide p53 añade **98.436 experimentos**. Datos: github.com/Minqi824/ADBench; paper: arxiv.org/abs/2206.09426. La p49 es solo la portada ("Comparativa").

**Tipos de anomalía (p50: cuatro scatters 2D, normales en azul y anomalías en rojo):**
- **(a) Local:** las anomalías están en el borde o dentro de los clusters normales; solo son raras respecto a su vecindario.
- **(b) Global:** anomalías dispersas entre los clusters, lejos de la masa normal.
- **(c) Dependency:** anomalías intercaladas en estructuras alargadas; rompen la relación entre variables, no la posición marginal.
- **(d) Clustered:** las anomalías forman su **propio grupo compacto**, aproximadamente en (47, −22), separado de los normales.

**Profesor.** La supervisión "la sabréis de forma automática: si tenéis etiquetas, unas pocas o ninguna". Los tipos de anomalía son "lo que hemos visto". Lo nuevo de evaluación es el ruido: "¿qué pasa si tenemos anomalías duplicadas, características completamente irrelevantes y errores de anotación?". "No espero que nadie se acuerde de esto más tarde; simplemente os dará una idea de qué funciona mejor y qué funciona peor."

*Fuente: slides p49, p50, p52, p53 · transcripción [03:44:13]–[03:45:02]*

---

## 11. ADBench por nivel de supervisión (slide p53)

**Idea.** Con los 57 datasets juntos, **las diferencias entre no supervisado, semi-supervisado y supervisado no son tan grandes**. Si hay pocas etiquetas, da bastante igual; las etiquetas sirven más para **validar**.

**Lectura de la slide.** Arriba, boxplots de AUCROC (%) por algoritmo; abajo, tiempo en segundos (escala log). Fondo amarillo = no supervisados, verde = semi-supervisados, lila = supervisados. Medianas leídas a ojo, con un error de ±3 puntos:
- **No supervisados:** PCA ~71, OCSVM ~66, LOF ~64, CBLOF ~71, COF ~63, HBOS ~71, KNN ~70, SOD ~68, COPOD ~72, ECOD ~73, **DeepSVDD ~53** (caja muy estrecha), **DAGMM ~58**, LODA ~62 e IForest ~72. Casi todos tienen bigotes desde 20–35 % hasta ~100 %: la variabilidad entre datasets es enorme.
- **Semi-supervisados:** GANomaly ~60, DeepSAD ~72, REPEN ~73, DevNet ~75, PReNet ~75, FEAWAD ~69 y **XGBOD ~77**.
- **Supervisados:** NB ~53, SVM ~56, MLP ~63, ResNet ~60, FTT ~71, RF ~58, LGB ~58, **XGB ~78** y **CatB ~81** (la mediana más alta).
- **Tiempo** (medianas aproximadas):
  - No supervisados: HBOS, PCA, COPOD, ECOD, LODA, IForest y LOF entre 0,2 y 0,6 s; OCSVM ~2 s; CBLOF ~1 s; KNN ~0,5 s; COF ~5 s (con bigote hasta ~500 s); DAGMM ~15 s; SOD ~25 s; DeepSVDD ~40 s.
  - Semi-supervisados profundos: 5–40 s; XGBOD ~40 s, con bigote hasta ~700 s.
  - Supervisados: NB/SVM/RF/LGB/XGB entre 0,15 y 0,5 s; MLP ~1–2 s; CatB ~3 s; ResNet ~50 s; FTT ~80 s, con bigote hasta ~3.000 s.
- (nota del redactor) La † marca los métodos de *ensemble* (LODA, IForest, XGBOD, RF, LGB, XGB, CatB). Las slides solo muestran AUCROC; el paper de ADBench también reporta AUCPR, que no se proyectó.

**Profesor.** "Tampoco hay tanta diferencia en función de si son no supervisados, semi-supervisados o supervisados… el hecho de que las anomalías no tengan nada que ver con vuestros datos hace que el hecho de que no tengan supervisión no afecte tanto… estamos intentando clasificar lo que tenemos contra el resto del mundo, y el resto del mundo no es fácil de caracterizar con 3 puntos. Ese es el primer mensaje: si yo tengo etiquetas y son muy pocas, lo más probable es que me dé bastante igual si es supervisado o no supervisado. Y, como decíais antes, las puedes utilizar para validación."

*Fuente: slide p53 · transcripción [03:45:28]–[03:46:10]*

---

## 12. ADBench por tipo de anomalía (slide p54)

**Idea.** **El tipo de anomalía importa mucho más que el nivel de supervisión.** "El performance va desde el 50 % hasta casi el 100 % en el mismo conjunto de problemas. Estos son promedios en muchas bases de datos, en 57. Entonces esto es mucho más importante que lo anterior. Si puedo saber qué tipo de problema tengo, bien; y si no, espera cualquier cosa entre casi aleatorio hasta que funcione perfecto."

**Lectura de la slide.** Eje X = % de anomalías etiquetadas (1, 5, 10, 25, 50, 75 y 100 %); eje Y = AUCROC (%); la línea discontinua es el **mejor no supervisado**. Para cada tipo, el panel izquierdo muestra los semi-supervisados y el derecho los supervisados. Valores a ojo, con un error de ±2–3 puntos.

| Tipo | Mejor no sup. | Semi-supervisados (1 % → 100 %) | Supervisados (1 % → 100 %) | Lectura |
|---|---|---|---|---|
| **(a) Local** | ≈93,5 | XGBOD 82→95 (lo supera solo con ≥50 % de etiquetas; al 25 % ≈92,5); DeepSAD 83→90; REPEN 77→84; FEAWAD 65→85; GANomaly plano ~79→78; PReNet 51→65; DevNet 51→60 | CatB 86→92; XGB 74→92; RF y LGB 63→92; FTT 60→87; ResNet 53→88; NB 48→86; SVM 68→91; MLP 52→79 | **Ningún supervisado alcanza al mejor no supervisado**; solo XGBOD con muchas etiquetas |
| **(b) Global** | ≈100 | XGBOD 96→100; DeepSAD 95→99,5; REPEN 94→99; FEAWAD 82→100; GANomaly ~94,5→95,5; PReNet y DevNet 70→93 | CatB ~99,5 desde el 1 %; FTT 89→99; XGB 92→99; RF 72→99; LGB 68→98; NB y SVM 66→99; ResNet 63→99; MLP <50 al 1 % → ~93,5 | Un no supervisado ya lo resuelve |
| **(c) Dependency** | ≈90 | XGBOD 78→95 (lo supera desde el 10 %); DeepSAD 83→91,5 (lo supera desde ~75 %); FEAWAD 63→87; REPEN 73→84; GANomaly 73→75; PReNet 51→70; DevNet 51→68 | CatB 86→96 (lo supera desde el 10 %); RF, LGB y XGB 60→94; FTT 58→90; ResNet ~63→~90; SVM 54→88; MLP 49→81; **NB ~47→57** | **Aquí las etiquetas sí ayudan** (boosting y XGBOD) |
| **(d) Clustered** | ≈96 | Casi todos 92–97 al 1 % y ~99–100 desde el 5–10 %; GANomaly 91→88,5→97,5 | **CatB ~99 desde el 1 %**; XGB ~95,5 y LGB ~95 al 1 %; NB, SVM y ResNet ~70; RF ~57; MLP <50; todos ≈100 con ≥50 % | Con un grupo compacto, **pocas etiquetas bastan** |

- (nota del redactor) Según el paper de ADBench (no se proyectó), los experimentos por tipo usan **anomalías sintéticas** generadas sobre datasets reales. Por eso p54 sirve para entender qué método encaja con qué tipo, pero sus números no son comparables con los de p53.

*Fuente: slide p54 · transcripción [03:46:18]–[03:47:11]*

---

## 13. ADBench: conclusiones para no supervisados y *no free lunch* (slide p55)

Citas literales de la slide:
- "None of the unsupervised methods is statistically better than the others."
- "Some DL-based unsupervised methods like DeepSVDD and DAGMM are surprisingly worse than shallow methods."
- "DL-based unsupervised algorithms are harder to train."
- "Future algorithms should be evaluated on large testbeds like ADBench for statistical tests."
- "The **no-free-lunch theorem**, suggests there is no universal winner for all tasks."

**Profesor.** "Eso os lo habrán dicho muchas veces, no free lunch… misma conclusión de siempre: si las suposiciones son correctas, todo bien; si son incorrectas, mala suerte." En la transcripción pone "si hacéis las funciones"; lo interpreto como *suposiciones*. "Este artículo vale la pena mirarlo porque contiene todos estos resultados" [03:47:48].

*Fuente: slide p55 · transcripción [03:47:20]–[03:47:53]*

---

## 14. ADBench con datos ruidosos y corruptos (slides p56–p57)

**Idea.** Hay tres corrupciones: **anomalías duplicadas**, **features irrelevantes** y **errores de anotación**. De esta última solo hay mención (en p52 y en la transcripción); no hay gráfica.

**Lectura de p56** (Δ AUCROC en puntos porcentuales respecto a los datos limpios; solo métodos no supervisados):
- **Datos duplicados** (X = factor de duplicación 2–6; Y entre −80 y +80):
  - Los dos peores con diferencia son **SOD** (−35 con ×2 → −48 → −58 → −66 → −73 con ×6) y **COF** (−36 → −45 → −50 → −54 → −58).
  - KNN: −8 → −27. LOF y CBLOF: ≈ −20 a −23 con ×6. ECOD, COPOD, PCA, IForest y LODA: −10 a −19 con ×6. OCSVM y HBOS: −5 a −10.
  - **DeepSVDD ≈ 0 (+2 → 0)**: es el único que no se ve afectado.
- **Features irrelevantes** (X = 1, 5, 10, 25 y 50 % de features irrelevantes; Y entre −15 y +15):
  - Hasta el 5 %, todos ≈ 0.
  - El más afectado es **DAGMM** (−1,3 → 0 → −4,7 → −6,8 → **−11,3**). Le siguen OCSVM y CBLOF (~−8 al 50 %); el resto está entre −3 y −5 al 50 %. HBOS ≈ −1 y LODA ≈ −3.
  - **DeepSVDD mejora** (+4,6 al 10 % y al 25 %, ~0 al 50 %); **COF** sube a +3,8 al 25 % y +1,8 al 50 %.

**Profesor.** Las escalas no son comparables: "esto va de 0 a 80 %, esto de 0 a ±10 más o menos". "En cuanto a datos duplicados hay más que nada métodos que [se rompen]." La transcripción dice "métodos de rehabilitar"; es una interpretación dudosa, coherente con la gráfica. "Si tenéis características irrelevantes, normalmente suele ser todo razonable… salvo que sean muchas, porque entonces los métodos de distancia fallan mucho: empiezan a medir distancias que no tienen nada que ver con la tarea, pero que aun así te sitúan muy lejos del resto de los datos." **Ejemplo:** una tarea resoluble con **2 features** a la que se añaden **500 de ruido**: "me distorsiona completamente cualquier análisis de distancia". Mientras el número de irrelevantes sea **comparable** al de relevantes no hay problema; al crecer, los problemas afectan "especialmente a los modelos que funcionan con distancia o probabilidad".

**Conclusiones literales (p57):**
- "All unsupervised methods are severely impacted by duplicated anomalies."
- "Unsupervised methods often assume the underlying data is imbalanced with only a smaller percentage of anomalies."
- "Incorporate unsupervised feature selection."

(nota del redactor) La frase "all… severely impacted" contradice en parte la propia gráfica, donde DeepSVDD apenas cambia. Por qué dañan los duplicados: al repetir anomalías se forman grupos densos que dejan de parecer raros en densidad o vecindad (pasan a comportarse como *clustered*). Eso rompe la hipótesis de escasez de la segunda conclusión y castiga sobre todo a los métodos de vecindario local (SOD, COF, KNN, LOF).

*Fuente: slides p52, p56, p57 · transcripción [03:44:29]–[03:44:47], [03:47:38]–[03:49:23]*

---

## 15. Etiquetas de anomalía: validar antes que entrenar *(tramos fuera del rango del borrador)*

**Idea.** Es un mensaje que el profesor repite en toda la clase: **si hay pocas anomalías etiquetadas, se usan para validar** (comprobar el detector y fijar umbrales) más que para entrenar.

**Citas.**
- "Si tenéis datos etiquetados de las anomalías, lo que sí podéis ver es cuántas de vuestras anomalías caen fuera de 2 sigma o de 3 sigma… y podéis ver si este sistema os funciona o no. Normalmente no tenemos etiquetas de anomalías, pero si las tuviésemos, lo mejor que se puede hacer con ellas, más que entrenar un modelo, es verificar que algo de este estilo está funcionando bien." [01:51:28]–[01:51:57]
- A la pregunta de un alumno sobre fijar el umbral con casos históricos: "Si yo tuviera anomalías y son muy pocas, nunca las utilizaría para entrenar el modelo; las utilizaría para validar las decisiones que tomaba con los umbrales." "Si tenéis tantas anomalías como datos buenos, entonces sí las puedes utilizar para entrenar… pero suele ser raro." Las anomalías, "hasta que no tienen más densidad y se convierten en su propia clase", pueden aparecer en cualquier sitio del espacio. [02:45:03]–[02:46:02]
- "Si las tenéis, las podéis utilizar para lo que queráis: o bien para validar, o bien para entrenar." [02:52:06]. La transcripción dice "ordenar"; lo interpreto como *entrenar*.
- Con ADBench: "si tengo etiquetas y son muy pocas… las puedes utilizar para validación" [03:46:00]–[03:46:10].
- Etiquetas **incompletas** en finanzas: "durante un tiempo alguien ha etiquetado en detalle, luego ha venido alguien que no sabía nada, luego alguien que ha puesto a Claude a etiquetar"; hay picos posteriores que "nadie los va a etiquetar nunca más" (semi-supervisado). [00:43:42]–[00:44:21]

(nota del redactor) Nuestro caso es intermedio: 295 fraudes no son "3 puntos", y ADBench (p54) muestra que con etiquetas abundantes el boosting gana en anomalías de dependencia y clustered. La lectura razonable es entrenar supervisado **y** reservar los fraudes de validación para elegir modelo y umbral sin tocarlos al entrenar. Además, `Class = 0` puede contener fraudes no detectados (etiquetas incompletas): eso baja artificialmente la precisión medida.

---

## 16. Desbalanceo: re-muestreo y SMOTE *(tramo fuera del rango del borrador)*

**Profesor.** La detección de anomalías puede verse como el caso extremo del desbalanceo ("2 imágenes de perro y 5.000 de gato"), y "cualquier cosa que habéis visto para clases muy desbalanceadas" vale:
- **Submuestrear** la clase mayoritaria ("quitar muestras hasta que está todo balanceado"). Problema: "eliminar información que ya tengo no es bueno, porque estoy tirando cosas a la basura".
- **Sobremuestrear** con ruido o muestras sintéticas, por ejemplo con la misma varianza que la clase. Problema: "tiene asunciones muy fuertes".
- **SMOTE**: "una combinación de las 2: coge puntos reales, los une e intenta muestrear en la zona que los une en función de la distancia que los separa".
- Advertencia clave: "siempre asumir que la distribución de los outliers va a ser una combinación de un millón de clases que nunca habéis visto… No tiene por qué ser que todos los outliers vengan de una nueva clase muy bonita."

(nota del redactor) Para la evaluación:
- El re-muestreo se aplica **solo dentro del fold de entrenamiento**, nunca antes de partir los datos.
- La AUPR se mide siempre sobre la distribución real (0,17 %).
- Re-muestrear no cambia el orden ideal de los scores. Como la AUPR solo depende del orden, el re-muestreo no ayuda per se: suele bastar con `class_weight` / `scale_pos_weight`.

*Fuente: transcripción [02:46:08]–[02:50:07]*

---

## 17. Dataset shift y monitorización en producción *(tramo fuera del rango del borrador)*

**Profesor.** El *dataset shift* son "modificaciones en los datos, tanto en los datos como en las etiquetas". Ejemplo hospitalario (alta en 24 h):
- **Shift en los datos (covariables):** el covid es una enfermedad nueva que "no estaba en mi población original"; la pirámide poblacional cambia lentamente.
- **Shift en las etiquetas:** cambia el criterio clínico (36,5 °C → rango 36,1–36,9 °C); "el punto no ha cambiado, pero las etiquetas sí".
- "En la práctica pasa un montón cuando tenéis un sistema en producción, a los de finanzas también": "se mira la salida de los modelos y se guarda un **conjunto de validación en producción** para ver la estimación del error del modelo, porque habrá momentos en que empiece a fallar."
- Qué hacer: (1) deshacer el movimiento con una transformación; (2) reentrenar y aceptar el cambio; (3) adaptar los datos antiguos o "simplemente reentreno y espero que todo vaya bien". Los periodos anómalos (criterios temporales en covid, crisis financieras) se marcan en el tiempo y no se usan como datos normales; "puede que necesites modelos específicamente para eso".
- Fuera de los dominios con cambios en el tiempo (finanzas, medicina), "lo que se suele hacer es ignorar que esto pasa… y se ponen alertas de vez en cuando para reentrenar".

**Relación con OOD.** Los outliers son "el caso extremo" del shift; se quieren quitar porque "si los utilizo en el entrenamiento me van a cambiar la frontera" y al evaluar meten ruido ajeno a la tarea [03:09:27]–[03:09:57].

*Fuente: transcripción [03:09:27]–[03:18:28]*

---

## 18. Uso práctico: logs, alertas, ciberseguridad y almacenamiento

- Un alumno tiene muchos registros de peticiones. Profesor: para logs "la gente sí lo utiliza; lo suelo hacer de forma **no supervisada**", como un **sistema de alertas para luego mirar más**. Antes se hacía con NLP; ahora con LLMs ya entrenados para que no sea costoso.
- **El coste real es almacenar**: "la mayoría del tiempo no teníamos alertas… se almacenaba, se almacenaba… y no había nunca nada"; es un problema "más de infraestructura".
- Herramienta: **One-Class SVM con bag of words**, "una cosa muy sencilla". Entrenar puede ser caro, pero con OCSVM **lineal** "solo tienes que guardar los pesos de las características": cada texto nuevo se multiplica por los pesos y ya está.
- Ciberseguridad y código: antes NLP, ahora LLMs, "y luego lo mismo: distancia o probabilidades más umbral". Después hay "gente que está de guardia" revisando las alertas.
- (nota del redactor) $f(\mathbf{x})=\mathbf{w}^\top\phi_{BoW}(\mathbf{x})-\rho$, con alerta si $f(\mathbf{x})<0$; la inferencia cuesta $O(\text{nº de términos})$. Es el mismo esquema operativo que el fraude: el analista revisa el top de alertas.

El resto del tramo ([04:25:35]–[04:30:47]) son despedidas y consejos sobre doctorados en Reino Unido (Cambridge, Oxford, Edimburgo, Manchester; título, CV, carta o propuesta, entrevista y normalmente dos supervisores). No tiene contenido técnico. El tramo [04:21:03]–[04:22:27] es el cierre del taller de deep learning: muestrear con la probabilidad de un modelo deep y "hacer fallar" el discriminador de una GAN.

*Fuente: transcripción [04:22:33]–[04:25:25]*

---

## 19. Métricas para desbalanceo extremo: por qué AUPR y no ROC

**En las fuentes.**
- **AUROC**: área bajo la curva TPR–FPR; es la probabilidad de que un positivo puntúe más que un negativo, y 0,5 equivale al azar (keshtmand24a p. 4; 1907.00811 §IV-A; OOD CF p. 6). Es la métrica de todas las gráficas de ADBench proyectadas.
- **AUPR**: área bajo la curva precisión–recall, con $P=TP/(TP+FP)$ y $R=TP/(TP+FN)$. **FPR@95 %TPR**, con $FPR=FP/(FP+TN)$ (keshtmand24a Ap. A.1).
- **Tasa de detección a FPR fijo (0,2)** (1907.00811, Fig. 4). **Umbral fijado en validación** dejando el 95 % de los ID por debajo (keshtmand24a §3.2).
- **Significación**: Wilcoxon signed-rank con $p<0{,}05$ y 4 u 8 semillas (papers de Keshtmand); ADBench pide testbeds grandes para hacer tests estadísticos (p55).
- En OOD CF, el AUROC se usa **al revés** (ID positivo, contrafactuales negativos): un valor bajo significa contrafactuales más realistas.
- Ejemplo real de que la AUPR depende de la proporción de positivos: con CIFAR100 como ID y SVHN como OOD (positivo), 1D Typicality da **AUROC 0,898 y AUPR 0,941** (keshtmand24a Tabla A.2). (nota del redactor) La AUPR supera al AUROC porque el conjunto OOD de SVHN es más grande que el ID; con 0,17 % de positivos pasa justo lo contrario.

**(nota del redactor) Por qué AUPR con 0,17 % de fraude.**
1. **El AUROC está inflado por los verdaderos negativos.** $FPR=FP/(FP+TN)$; con ~170.000 negativos, 500 falsos positivos son solo un 0,3 % de FPR. Un modelo puede tener AUROC 0,95 (el baseline del profesor tiene 0,948) y aun así enterrar cada fraude entre decenas de falsas alarmas. La precisión no tiene $TN$ en el denominador y refleja directamente la carga del analista.
2. **Baseline distinto.** Un score aleatorio da AUROC = 0,5 pero **AUPR = prevalencia** = $295/170.884\approx0{,}0017$. El 0,567 del profesor es unas **330 veces el azar**. Por eso las AUPR no se comparan entre datasets con prevalencias distintas, y si la tasa de fraude en producción difiere, la AUPR también cambiará con el mismo modelo.
3. **Qué calcula `average_precision_score`.** $AP=\sum_n (R_n-R_{n-1})\,P_n$, la media de la precisión en la posición de cada positivo, sin interpolación trapezoidal (la trapezoidal sobreestima con pocos positivos). Por tanto:
   - **Relación con precision@k.** AP = media de precision@$k_i$, donde $k_i$ es el rango del $i$-ésimo fraude. Premia sobre todo colocar los fraudes **en lo más alto** del ranking, que es exactamente lo que revisa un equipo antifraude. Conviene reportar también **precision@k** y **recall@k** para $k$ operativos (p. ej. $k$ = 100, 500, 1.000).
   - Solo depende del **orden**: es invariante a transformaciones monótonas. El CSV no necesita scores calibrados, solo bien ordenados (mayor = más fraude). Para combinar modelos hay que promediar rangos o scores normalizados, porque las escalas brutas no son comparables.
   - Los empates penalizan: scores muy discretos (p. ej. HBOS) pueden perder AP.
4. **AUROC como métrica secundaria.** Sirve para comparar con ADBench y con la literatura, pero **las decisiones se toman con AUPR**.

---

## 20. Estimar la AUPR con pocos positivos *(nota del redactor)*

- Con el split del profesor (70/30 estratificado, seed 42), el test tiene ≈ **89 fraudes** frente a ~51.000 normales. La AP es muy sensible a dónde caen unos pocos fraudes: subir o bajar 3–4 fraudes en el ranking mueve la AP varias centésimas.
- **Orden de magnitud por simulación** (scores binormales, 89 positivos, 51.265 negativos, 300 réplicas): con una AP media ≈ 0,47 la desviación típica es ≈ **0,054** (IC95 % ≈ 0,37–0,57); con una AP media ≈ 0,63, la desviación es ≈ 0,049 (IC95 % ≈ 0,54–0,73). El AUROC, en cambio, varía solo ±0,01. **Conclusión: diferencias de AUPR menores de ~0,05 en un único split no son concluyentes.**
- Cómo hacerlo bien:
  1. **CV estratificada repetida** (`RepeatedStratifiedKFold`, 5 folds × 3–5 repeticiones; ~59 fraudes por fold). Se reporta media ± desviación de la AP por fold. La AP *pooled* (concatenando las predicciones out-of-fold) es más estable, pero solo es válida si los scores de distintos folds son comparables en escala; si no, mejor la media por fold.
  2. **Intervalos bootstrap** sobre el test: se remuestrea estratificado (positivos y negativos por separado) unas 1.000–2.000 veces y se toma el percentil 2,5–97,5 de la AP.
  3. **Comparar modelos de forma pareada**: mismo fold y mismas semillas; diferencia de AP por fold con test de Wilcoxon (como en los papers) o bootstrap pareado de la diferencia.
  4. Cada pipeline completo (escalado, selección de features, re-muestreo, umbral) se ajusta **dentro** del fold; si hay muchas configuraciones, **CV anidada** o un hold-out final que no se toque hasta el final.
  5. Los modelos no supervisados entrenados "solo con normales" (como el GMM del profesor) se ajustan con los normales del fold de entrenamiento y se puntúan sobre el fold de validación completo, con normales y fraudes.

---

## 21. Riesgos de evaluación: duplicados y data shift *(nota del redactor, apoyada en §14 y §17)*

**Duplicados (4.167 filas duplicadas en construcción).**
- **Fuga train/test**: si una fila idéntica cae en train y en test, los modelos que memorizan (KNN, árboles profundos, boosting) la aciertan "gratis" y la AUPR sale optimista. Esto es especialmente grave si hay fraudes duplicados.
- Solución: agrupar las filas idénticas (hash de todas las columnas) y partir con **`StratifiedGroupKFold`**, para que cada grupo quede entero en un solo lado. También se puede deduplicar antes de partir y reportar ambas cifras.
- Efecto sobre los no supervisados (ADBench p56): las anomalías duplicadas hunden SOD, COF, KNN y LOF. Conviene entrenar los detectores de densidad o vecindario con datos deduplicados (o con pesos), aunque luego se puntúen todas las filas.
- Comprobar también duplicados **entre construcción y producción**. Si existen, son información legítima (misma transacción repetida); decidir y documentar cómo se tratan.

**Data shift construcción → producción.**
- Indicios a revisar: distribución por feature (test KS, PSI, comparar medias y cuantiles de X1..X20 y Amount), tasa de outliers según el detector no supervisado, y distribución del score final en ambos conjuntos.
- **Adversarial validation**: etiquetar construcción = 0 y producción = 1 y entrenar un clasificador (LightGBM) con CV. Un **AUROC ≈ 0,5** indica que no hay shift detectable. Un AUROC ≫ 0,5 indica shift: las importancias del clasificador señalan qué features se han movido. Las filas de construcción con más probabilidad de "parecer producción" forman un conjunto de validación más realista.
- Si hay shift:
  - Preferir modelos y features robustos (quitar las que más se mueven si aportan poco) y normalizar por rangos.
  - Reforzar el componente no supervisado de densidad (GMM, IForest), que cubre fraudes "nuevos" fuera de lo visto, como argumentan el paper OOD CF y el profesor sobre que los outliers vienen de "un millón de clases que nunca habéis visto".
  - Ojo con el *transductive fitting*: ajustar los escaladores o los detectores no supervisados con datos de producción es legítimo (no usa etiquetas), pero hay que documentarlo.
- La prevalencia en producción puede no ser 0,17 %: eso cambia la AUPR obtenida aunque el ranking sea igual de bueno.

---

## 22. Qué probar primero en nuestro caso: ADBench ordenado *(nota del redactor)*

| Prioridad | Qué | Por qué (fuente) | Riesgo / cuándo no |
|---|---|---|---|
| 1 | **Boosting supervisado** (CatBoost, XGBoost, LightGBM) con `scale_pos_weight` / `class_weight` | Las medianas más altas en p53 (CatB ~81, XGB ~78); domina en *dependency* y *clustered* con etiquetas (p54); rápido | En anomalías *local* no supera al mejor no supervisado (p54-a); se sobreajusta a los fraudes conocidos |
| 2 | **Baselines no supervisados rápidos**: GMM del profesor, IForest, HBOS, ECOD/COPOD, PCA (reconstrucción) y Mahalanobis | Los mejores no supervisados de p53 (~71–73) y muy baratos; ninguno es estadísticamente mejor (p55): probar varios | Se hunden con duplicados (p56) → entrenar deduplicados; HBOS da muchos empates |
| 3 | **Híbrido tipo XGBOD**: los scores no supervisados como features del boosting | XGBOD es el mejor semi-supervisado (p53 ~77) y supera al mejor no supervisado en *local* (≥50 % de etiquetas) y *dependency* (≥10 %) | Los scores no supervisados se calculan dentro del fold (fuga si se ajustan con todo) |
| 4 | **Vecindario** (KNN, LOF) con features escaladas (log1p(Amount) + estandarizar) | Útil si los fraudes son *local*; LOF está pensado para eso | Muy sensibles a duplicados (p56) y a features irrelevantes (§14, 2 vs 500) |
| 5 | **Ensemble por rangos** (media de rangos de 1–3) | Ningún método gana siempre (no free lunch, p55); el tipo de anomalía es incierto | Ganancias pequeñas: verificar con CV pareada (§20) |
| 6 (último) | **Deep no supervisado** (autoencoder, DeepSVDD, DAGMM) y deep supervisado (MLP, ResNet, FTT) | DeepSVDD y DAGMM, "surprisingly worse than shallow methods" y "harder to train" (p55); FTT es lento (p53) | Solo como candidato de diversidad para el ensemble |
| Siempre | **Diagnosticar el tipo de anomalía** antes (01_EDA): PCA/UMAP 2D, como en p50 | "Si algo se puede visualizar, se visualiza" (§8); el tipo pesa más que la supervisión (§12) | — |

---

## 23. Herramientas: sklearn y PyOD (chuleta de API)

| Método | Clase | Parámetros clave | Cómo sacar el score | ¿Mayor = más anómalo? | Escalabilidad (170k×21) |
|---|---|---|---|---|---|
| Regresión (residuo) | `sklearn.linear_model.LinearRegression` | — | `np.abs(y - model.predict(x))` | Sí | Excelente |
| Regresión polinómica | `PolynomialFeatures(degree)` + `LinearRegression` | `degree` | `np.abs(y - y_hat2)` | Sí | Buena (el nº de features crece con el grado y d) |
| PCA (reconstrucción) | `sklearn.decomposition.PCA` | `n_components` | `((X - pca.inverse_transform(pca.transform(X)))**2).sum(1)` o `np.abs(X_pca[:, k:]).sum(1)` | Sí | Excelente |
| K-means / VQ (nota red.) | `sklearn.cluster.KMeans` / `MiniBatchKMeans` | `n_clusters`, `init`, `n_init` | `km.transform(X).min(axis=1)` | Sí | Buena |
| GMM | `sklearn.mixture.GaussianMixture` | `n_components`, `covariance_type` (spherical/diag/tied/full), `max_iter`, `random_state`; `bic()` para elegir K | `score_samples(X)` = log p(x) | **No** → `-score_samples` | Buena |
| DBSCAN | `sklearn.cluster.DBSCAN` | `eps` (radio), `min_samples` (n) | `labels_ == -1` (solo etiqueta) | Binario; sin score continuo ni `predict` | Mala |
| LOF | `sklearn.neighbors.LocalOutlierFactor` | `n_neighbors` (k), `contamination`, `novelty` | train: `negative_outlier_factor_`; nuevos (`novelty=True`): `score_samples` | **No** → `-negative_outlier_factor_` / `-score_samples` | Mala ("escalan muy mal") |
| Isolation Forest | `sklearn.ensemble.IsolationForest` | `n_estimators`, `max_samples`, `max_features`, `contamination`, `warm_start`, `random_state` | `score_samples(X)` ("the lower, the more abnormal"); `decision_function`; `predict` → −1/1 | **No** → `-score_samples` | Muy buena |
| One-Class SVM (nota red.) | `sklearn.svm.OneClassSVM`; escalable: `Nystroem` + `sklearn.linear_model.SGDOneClassSVM` | `kernel`, `nu`, `gamma` | `decision_function` / `score_samples` | **No** → negar | Exacta: mala (O(n²–n³)); Nystroem+SGD: buena |
| ECOD (PyOD) | `pyod.models.ecod.ECOD` | `contamination` (0.1), `n_jobs` | `decision_function(X)`; en train `decision_scores_` | **Sí** | Excelente |
| Otros PyOD (nota red.) | `pyod.models.{iforest.IForest, lof.LOF, knn.KNN, ocsvm.OCSVM, gmm.GMM, pca.PCA, copod.COPOD, hbos.HBOS, auto_encoder.AutoEncoder}` | `contamination` + los propios | `decision_function` / `decision_scores_` | **Sí** (toda PyOD) | HBOS/COPOD/ECOD/PCA muy rápidos; KNN/LOF/OCSVM lentos |

**Signo y umbral** (comprobado en sklearn 1.8):
- sklearn: `predict` devuelve **−1 = outlier, 1 = inlier**; `score_samples` sigue la convención "mayor = más normal". El profesor lo resuelve con el signo menos, `y_train_scores = -clf.score_samples(X)` (celda 7, "The lower, the more abnormal").
- `decision_function = score_samples - offset_` en IsolationForest, LOF (novelty) y OneClassSVM. Es un desplazamiento constante, así que **el ranking, y por tanto AUPR/ROC, es idéntico** con una u otra. `offset_` depende de `contamination`: con `'auto'` vale −0,5 en IF y −1,5 en LOF.
- PyOD: `decision_function` ya da mayor = más anómalo; `labels_`/`predict` dan **1 = outlier, 0 = inlier**; `contamination` solo fija el umbral. (nota del redactor) `pyod.models.iforest.IForest` envuelve el IsolationForest de sklearn y devuelve su `decision_function` cambiada de signo.

---


Detalle de cada algoritmo (teoría, código del profesor, ejercicios): ver `01_metodos_clasicos.md`, parte B.

## Resumen de los papers (problema · método · datos · métricas · resultados · mensaje para nosotros)

### A. *Counterfactual generation for Out-of-Distribution data* (Keshtmand, Santos-Rodriguez, Lawry)
- **Problema.** Explicar **por qué** un punto se clasifica como OOD. Los contrafactuales estándar fallan porque el contrafactual sigue siendo OOD.
- **Método.** "OOD CF": latentes con PCA → partición en discriminativas $z_d$ (top-k por selección secuencial con QDA, k = la mitad) y no discriminativas $z_n$ → descenso de gradiente en dos etapas minimizando $-\log p$ bajo un GMM (1 componente para $z_n$, nº de clases para $z_d$). Recorrido: far-OOD → near-OOD → ID.
- **Datos.**
  - 2D sintético: ID en (3,0) y (−3,0) con $\Sigma=0{,}5I$; OOD en (0,2) con $\Sigma=0{,}3I$. La trayectoria se mueve primero en pc2 (no discriminativa) y luego en pc1.
  - Tabular: Wine (la clase 2 es OOD), Pima Diabetes (edad por encima del cuartil superior = OOD) y Thyroid (clase "subnormal" = OOD).
  - Imágenes: MNIST↔KMNIST y MNIST→ColoredMNISTRed.
- **Métricas.** Pérdidas Non-dis y Dis (NLL), L1 (minimalidad) y AUROC entre ID y contrafactuales con Mahalanobis (**más bajo = mejor**). Media de 4 semillas, Wilcoxon.
- **Resultados** (Non-dis / Dis / L1 / AUROC; baselines con Alibi sobre una red de 2 capas ocultas, 500 épocas, SGD, batch 128, lr 0,01):
  - **Wine:** OOD CF 3,58\* / 5,36\* / 7,64 / **0,34\***; Proto 8,57 / 7,09 / 7,01 / 0,99; CFI 123,18 / 72,88 / 9,42 / 1,00; CEM 24,34 / 14,54 / 7,80 / 1,00.
  - **Diabetes:** OOD CF 4,28 / 3,66 / **2,67\*** / **0,33\***; Proto 4,32 / 3,15 / 4,76 / 0,50; CFI 291,01 / 150,88 / 6,26 / 0,65; CEM 8,24 / 5,47 / 5,68 / 0,61.
  - **Thyroid:** OOD CF 3,22\* / 3,90\* / 4,73\* / **0,39\***; Proto 3,47 / 4,14 / 9,01 / 0,82; CFI 17,36 / 95,87 / 10,30 / 0,90; CEM 11,73 / 55,15 / 9,66 / 0,95.
  - **Imágenes (AUROC):** MNIST–KMNIST 0,26\* frente a 0,40–0,41; MNIST–CMNIST 0,52\* frente a 0,65–0,66; KMNIST–MNIST 0,22\* frente a 0,28–0,36. La L1 de OOD CF es menor (≈ 34–43 frente a ≈ 51–59).
  - Sin minimizar L1 explícitamente, OOD CF da menor L1 (salvo en Wine): "es más fácil mover el dato de OOD a ID que cruzar la frontera de una clase ID concreta".
  - Cualitativo: la 1.ª etapa aclara el fondo y la 2.ª añade o quita trazos (0, 6, 9). Con ColoredMNIST el dígito rojo apenas cambia: el método añade rasgos más que modificar los existentes. Con KMNIST como ID convendría un extractor neuronal en vez de PCA.
- **Mensaje para nosotros.** Para justificar un fraude muy anómalo no vale "la transacción legítima más cercana". Hay que explicar (1) en qué features se aleja de lo normal y (2) qué la separa de la clase. El GMM con NLL del baseline del profesor es el mismo estimador de densidad del paper, así que su $-\log p$ por componente es una explicación natural.

### B. *Typicality-based point OOD detection with contrastive learning* (Keshtmand, Santos-Rodriguez, Lawry — NLDL 2024)
El profesor lo describe así: "de hace 3 o 4 años… utilizamos la distancia de Mahalanobis… en combinación con una red neuronal… coge muchos de los ingredientes que hemos utilizado hoy… funciona más o menos bien en problemas que nos hemos inventado" [03:40:12]–[03:40:53].
- **Problema.** Los métodos de tipicalidad detectan bien **grupos** de OOD, pero no **puntos** sueltos.
- **Método: "1D Typicality".**
  1. Encoder ResNet-50 con **Supervised Contrastive loss**; salida de 128 dimensiones normalizada L2.
  2. El "grupo" de un punto son sus **K vecinos** en el latente, tomados entre los datos de test: no hace falta guardar el train, lo que ayuda a la privacidad.
  3. **Mahalanobis 1D** por dimensión propia y clase: $MD^{(d)}(z)=l_d^2/\lambda_d$. Score:
  $$D_{Typical}=\min_i\sum_{d}\sum_{l=1}^{K}\Big[\tfrac{MD^{(d,i)}(z_l)-\mu_{MD(d,i)}}{\sigma_{MD(d,i)}}\Big]^2$$
  4. Umbral: el 95 % de la validación ID queda por debajo.
- **Datos.** ID: CIFAR100, Caltech256, TinyImageNet. OOD: esos mismos más CIFAR10, SVHN y MNIST. Entrenamiento: 300 épocas, batch 256, SGD lr 3e−2, K = 10.
- **Métricas.** AUROC (OOD como positivo, 8 semillas), AUPR y FPR@95 %TPR; Wilcoxon.
- **Resultados** (AUROC):
  - Contrastivo frente a cross-entropy, con CIFAR100 como ID: →SVHN 0,898 frente a 0,473; →MNIST 0,770 frente a 0,422. Con TinyImageNet como ID: →MNIST 0,989 frente a 0,964, pero →SVHN 0,451 frente a 0,587 (aquí gana la CE).
  - Frente a KNN+ y SSD+ con el mismo encoder: gana en la mayoría de pares.
  - Frente a MSP, ODIN y MD: gana en la mayoría. ODIN gana en Caltech256→MNIST (0,912) y MD en TinyImageNet→SVHN y →Caltech256.
  - Vecinos (Ap. A.4): los ID tienen ~2–12 % de vecinos OOD y los OOD ~90–99 %.
  - Con Caltech256 como ID, muchos AUROC están cerca de 0,5.
- **Mensaje para nosotros.** (i) Mahalanobis es un score fuerte y simple. (ii) Los vecinos en producción aportan contexto (cuidado con duplicados y con anomalías clustered). (iii) La representación importa más que el score final. (iv) Ejemplo real de AUPR > AUROC cuando los positivos son mayoría (§19).

### C. *Location Anomalies Detection for Connected and Autonomous Vehicles* (Wang, …, Santos-Rodríguez, Piechocki — 2019)
- **Problema.** Vehículos conectados (V2X) que reportan una **posición falsa** ("transmisor fantasma" T′), ya sea por hackeo o por un sensor defectuoso. Contexto del profesor (no está en el paper): según lo que pagas al operador de red, tus paquetes se pierden antes, lo que es éticamente problemático; es un escenario ideal para ataques adversarios ("intento que haya problemas de clasificación y los coches chocan").
- **Método.** **Autoencoder profundo** no supervisado de 7 capas densas, con latente de dimensión 20. Entrada: posición del receptor, RSSI y posición reportada (dimensión 5). Se entrena **solo con datos sin anomalías** minimizando $\lVert X-X'\rVert^2$ (lr 0,00095). Los datos normales se parten 80/20 entre train y validación. **Score = error de reconstrucción.** Se valida con **AMI** entre las pérdidas de train y validación (AMI = 1,0; medias 4,11e−5 y 5,37e−5). El benchmark es un One-Class SVM lineal.
- **Datos.** Simulación OMNeT++/INET/SUMO de 2 km × 2 km del centro de Bristol, con 150 vehículos durante 1.800 s. Ocho datasets AD1–AD8 según la distancia T–T′ ([0,10) … ≥500 m). AD9: >30 m pero con la misma distancia al receptor (±1 m), es decir, cambia la dirección. AD10: diferencia de 10–20 m.
- **Métricas.** ROC/AUC y tasa de detección a **FPR = 0,2**.
- **Resultados.** Por debajo de 10 m, DAE y SVM ≈ aleatorio. Hacia 20 m el DAE mejora claramente; por encima de 30 m detecta bien con FPR baja; desde 100 m el DAE ≈ 100 %, mejor que el OCSVM. En AD9 el DAE empeora y el OCSVM ≈ aleatorio: "la dirección no se aprende por falta de features" → hacen falta más fuentes o sensores.
- **Mensaje para nosotros.** (i) "Entrenar solo con normales y puntuar por error" es la misma filosofía que el GMM del profesor. (ii) **Lo que no está en las features no se detecta**: los fraudes sutiles serán casi aleatorios para un no supervisado. (iii) Evaluar **por tramos de dificultad** (aquí por distancia; nosotros por Amount o por cuantil de score) muestra dónde falla el modelo. (iv) Son resultados "en problemas que nos hemos inventado".

---

## Lo que dijo el profesor (recomendaciones explícitas)

**Explicabilidad**
- Explicar por qué algo es outlier: como mínimo, el umbral ("umbral 5, distancia 6") y qué features generan la distancia [03:24:27]–[03:24:55].
- Para explicar outliers, dos etapas: acercar a una zona con densidad y después buscar el contrafactual [03:27:46]–[03:28:20].
- No usar LIME, SHAP ni contrafactuales directos sobre outliers; primero llevar el punto a la clase [03:33:16]–[03:34:17].
- Con el AI Act se puede "justificar" con contrafactuales absurdos (+10.000 €, −20 años). O se detecta el outlier y no se explica, o se explica de otra manera [03:34:30]–[03:36:01].

**Cómo leer resultados y elegir método**
- Los benchmarks sintéticos (MNIST/KMNIST) sirven para evaluar, no para el mundo práctico [03:37:14]–[03:37:40].
- Escepticismo: "los números en negrita son los del método propuesto" [03:40:58]; "los ejes empiezan y acaban donde cada uno quiere" [03:46:27].
- "Muchas métricas, pero unas más relevantes que otras según el problema"; "si algo se puede visualizar, se visualiza"; lo primero es entender en qué caso estamos [03:42:04]–[03:42:41].
- Más reciente no significa mejor [03:45:07]. Todos los métodos de ADBench son mejores que el azar [03:46:31].
- Con pocas etiquetas, supervisado ≈ no supervisado: usar las etiquetas para validar [03:46:00]–[03:46:10], [01:51:46], [02:45:03]–[02:45:19].
- El tipo de anomalía importa mucho más que la supervisión (del 50 % al 100 %); si no sabes cuál es, "espera cualquier cosa" [03:46:34]–[03:47:11].
- No free lunch [03:47:24]. Vale la pena mirar el paper de ADBench [03:47:48].

**Datos y producción**
- Duplicados: algunos métodos se rompen. Features irrelevantes: no hay problema salvo que sean muchas frente a las relevantes (distancia y probabilidad) [03:48:01]–[03:49:23].
- Desbalanceo: submuestreo, sobremuestreo o SMOTE, asumiendo que los outliers vienen de "un millón de clases que nunca habéis visto" [02:48:07]–[02:50:04].
- En producción (también en finanzas): guardar un conjunto de validación en producción, vigilar el shift en datos y etiquetas, y reentrenar [03:13:54]–[03:16:35].
- Logs: no supervisado, sistema de alertas, OCSVM lineal con bag of words; el cuello de botella es el almacenamiento [04:23:02]–[04:24:33]. Ciberseguridad: LLM + distancia o probabilidad + umbral [04:25:12]–[04:25:22].

---

## Aplicación a nuestra práctica (fraude, AUPR, 0,17 %)

- **Régimen de etiquetas.** Tenemos 295 fraudes etiquetados (el "100 %" de p54) y 170.884 filas. No es el caso de "3 puntos" del profesor, así que el supervisado tiene sentido; aun así, sus consejos se mantienen: **las etiquetas también son el instrumento de validación** (§15) y la evaluación debe ser honesta con tan pocos positivos (§20).
- **Plan de modelos.** Seguir la tabla de §22 y su reparto por notebooks:
  - 02: no supervisados, con el GMM del profesor como referencia (AUPR 0,567).
  - 03: boosting.
  - 04: híbrido tipo XGBOD y ensemble por rangos.
  - 99: el mejor según la AUPR por CV, con su intervalo.
- **Diagnóstico del tipo de anomalía (01_EDA).** PCA/UMAP 2D de fraude frente a no fraude:
  - Si los fraudes forman un cluster → supervisado con pocas etiquetas.
  - Si están dispersos (*global*) → IForest/KNN.
  - Si están pegados a los clusters normales (*local*) → LOF, donde el no supervisado compite.
  - Si rompen correlaciones entre X1..X20 (*dependency*) → Mahalanobis, reconstrucción PCA y boosting.
  - Como X1..X20 ya son tipo PCA (casi incorrelados en los normales), conviene mirar si en los fraudes aparecen correlaciones: es una señal de anomalía de dependencia.
- **Preprocesado.** Pasar Amount por `log1p` y estandarizar, para que no domine las distancias. Con 21 features, el riesgo de "2 frente a 500" es bajo, pero se puede probar a quitar features con distribución idéntica en fraude y no fraude para los métodos de distancia (p57: "incorporate unsupervised feature selection").
- **Duplicados.** `StratifiedGroupKFold` por hash de fila; detectores de densidad o vecindario entrenados deduplicados (§21).
- **Shift.** Adversarial validation construcción frente a producción en 01_EDA; documentar el resultado en el notebook (§21).
- **Métrica.** Decidir con AUPR (baseline ≈ 0,0017). Reportar AUROC por comparabilidad, más precision@k y recall@k. Para el CSV basta un score bien ordenado; al combinar modelos, promediar rangos.
- **Explicación en 99_ENTREGA.**
  1. Justificar las familias elegidas con ADBench (supervisión frente a tipo de anomalía; "más reciente no es mejor"; no free lunch).
  2. Explicabilidad: SHAP para el boosting sobre fraudes parecidos a los conocidos. Para los fraudes que levanta el componente de densidad, explicar con umbral + contribución por feature (z-scores o términos de Mahalanobis/GMM), no con contrafactuales directos (§3–§6).
  3. Reflexión: features anonimizadas, fraudes "cercanos a lo normal" casi indetectables (como AD1 del paper de vehículos), posible shift y etiquetas posiblemente incompletas. En la práctica, el sistema funcionaría como alertas con revisión humana, donde precision@k es la carga de trabajo del analista.

---

## Checklist de evaluación para el grupo

**Antes de modelar**
- [ ] Contar duplicados exactos (globales, entre fraudes, entre construcción y producción) y decidir el tratamiento.
- [ ] Crear `group_id` = hash de la fila y usar `StratifiedGroupKFold` (5 folds × 3 repeticiones) para todo el desarrollo.
- [ ] Mantener el split del profesor (70/30 estratificado, seed 42) **solo** para comparar con su 0,567 / 0,948.
- [ ] Visualizar fraude frente a no fraude (PCA/UMAP 2D, histogramas por feature) y anotar qué tipo de anomalía parece (local, global, dependency o clustered).
- [ ] Adversarial validation construcción frente a producción (AUROC del clasificador ± desviación) y test KS/PSI por feature.

**Durante el desarrollo**
- [ ] Ajustar todo el pipeline dentro del fold: escalado, selección, re-muestreo, detectores no supervisados (incluidos los scores que alimentan el híbrido) y umbral.
- [ ] Entrenar los no supervisados "solo con normales" usando los normales del fold de entrenamiento; puntuar el fold de validación completo.
- [ ] Usar como métrica principal `average_precision_score(y, score)` con score mayor = más fraude; comprobar el signo (p. ej. `-score_samples`).
- [ ] Reportar por modelo: AP media ± desviación entre folds, AUROC, precision@k y recall@k (k = 100, 500), y el baseline (prevalencia).
- [ ] Comparar los candidatos de forma pareada (mismos folds): Wilcoxon o bootstrap de la diferencia. No declarar ganador con ΔAP < ~0,05 sin test.
- [ ] Comprobar la estabilidad del ranking del score frente a semillas.

**Antes de entregar**
- [ ] Intervalo bootstrap estratificado (≥1.000 réplicas) de la AP del modelo final en un hold-out no usado para elegir.
- [ ] Reentrenar el modelo final con toda la construcción; puntuar producción.
- [ ] Revisar el CSV: 113.923 filas, una sola columna, sin NaN ni infinitos, mismo orden que producción, mayor = más fraude.
- [ ] Comparar la distribución de scores entre construcción (out-of-fold) y producción: un desplazamiento fuerte apunta a shift; comentarlo en la reflexión.
- [ ] En el notebook: justificación de las familias con ADBench, tabla de resultados con intervalos, explicación de algunos fraudes (SHAP y contribuciones por feature) y limitaciones.

---

## Cobertura de fuentes

| Fuente | Rango | Cubierto | Notas |
|---|---|---|---|
| slides_clasicas/p49.png | portada | sí | Solo el título "Comparativa" (§10). |
| slides_clasicas/p50.png | tipos de anomalía | sí | Cuatro scatters (§10). |
| slides_clasicas/p51.png | timeline | sí | 14 / 7 / 9 algoritmos con autores; años aproximados (§9). |
| slides_clasicas/p52.png | diseño ADBench | sí | 57 datasets, 30 algoritmos, 3 ángulos con secciones (§10). |
| slides_clasicas/p53.png | boxplots AUCROC y tiempo | sí | Medianas de los 30 métodos y tiempos corregidos (§11). |
| slides_clasicas/p54.png | AUCROC frente a % de etiquetas por tipo | sí | 8 paneles, en tabla y corregidos (§12). |
| slides_clasicas/p55.png | conclusiones no supervisados | sí | Literal (§13). |
| slides_clasicas/p56.png | duplicados / irrelevantes | sí | Valores corregidos (§14). |
| slides_clasicas/p57.png | conclusiones ruido | sí | Literal, con un matiz sobre DeepSVDD (§14). |
| transcripción 1940–2006 | [03:24:03]–[03:30:24] | sí | §1–§4. |
| transcripción 2007–2065 | [03:30:31]–[03:36:06] | sí | §5–§6. |
| transcripción 2066–2090 | [03:36:11]–[03:38:33] | sí | §7. |
| transcripción 2091–2118 | [03:38:38]–[03:41:10] | sí | Papers B y C; escepticismo (§8). |
| transcripción 2119–2152 | [03:41:14]–[03:44:04] | sí | §8–§9; el ejercicio "2019 en 5 min" pertenece al bloque deep. |
| transcripción 2153–2219 | [03:44:13]–[03:50:01] | sí | §10–§14; las últimas líneas son la transición a deep learning. |
| transcripción 2522–2536 | [04:21:03]–[04:22:27] | sí | Cierre del taller deep; sin contenido de evaluación (§18). |
| transcripción 2537–2565 | [04:22:33]–[04:25:25] | sí | §18. |
| transcripción 2566–2601 | [04:25:35]–[04:30:47] | sí | Despedida y doctorados (§18, una línea). |
| transcripción (otros tramos) | [00:43:42]–[00:44:21], [01:51:28]–[01:51:57], [02:45:03]–[02:52:06], [03:09:27]–[03:18:28], [04:19:38] | sí (nuevo) | Etiquetas para validar (§15), desbalanceo y SMOTE (§16), dataset shift y validación en producción (§17), visualizar (§8). Localizados con Grep de "etiqueta", "validación", "umbral", "desbalance" y "métrica". |
| papers/45_Counterfactual_generation_f.txt | completo | sí | Método, ecuaciones, tablas 1–2 verificadas, apéndices. |
| papers/keshtmand24a.txt | completo | sí | Método, métricas (A.1), tablas A.2–A.4 verificadas en los valores citados. |
| papers/1907.00811v1.txt | completo | sí | Arquitectura, validación AMI (Tabla I verificada), AD1–AD10. |

**Correcciones de esta revisión respecto al borrador**
- **p53 (tiempos):** COF no tarda 10–100 s, sino ~5 s de mediana (bigote ~500 s). Se añaden OCSVM ~2 s, MLP ~1–2 s, CatB ~3 s y XGBOD ~40 s; FTT llega a ~3.000 s. Se completan las medianas de DeepSAD, REPEN, DevNet, PReNet, FEAWAD y de todos los no supervisados.
- **p54:**
  - (a) Local: XGBOD supera al mejor no supervisado solo con ≥50 % de etiquetas (al 25 % ≈ 92,5 < 93,5).
  - (b) Global: el MLP empieza por debajo de 50 (no en ~60) y el ResNet llega a ~99 (no a ~93). Faltaba FTT (89→99).
  - (c) Dependency: DeepSAD solo supera la línea con ≥75 % de etiquetas. Se añade ResNet.
  - (d) Clustered: el MLP empieza por debajo de 50.
- **p56:** la mejora de COF solo se da al 25 % (+3,8), mientras que DeepSVDD mejora +4,6 al 10 % y al 25 %. Se señala que la conclusión de p57 ("all… severely impacted") choca con DeepSVDD ≈ 0.
- **Afirmación no verificable eliminada:** que el paper de tipicalidad usa conjuntos ID/OOD "10.000/10.000 equilibrados". Se sustituye por el dato real (SVHN: AUPR 0,941 > AUROC 0,898).
- **Transcripción:** "a la hora de evaluar es partido de ayer" se lee como "como vimos ayer" (las métricas se dieron en la sesión anterior). Se añade "todos funcionan mejor que aleatorio" [03:46:31]. "Métodos de rehabilitar" queda marcado como interpretación dudosa. La cita del paper B empieza en [03:40:12], no en [03:40:07].

**Huecos y dudas pendientes**
- La sesión de métricas ("como vimos ayer") no está en esta transcripción: puede haber criterios del profesor sobre AUPR o ROC que no recogemos.
- Los valores de p53, p54 y p56 son lecturas a ojo (±2–3 puntos). Para cifras exactas, ver las tablas del paper de ADBench (arXiv 2206.09426), que no está entre las fuentes.
- Los errores de anotación (ángulo III) no tienen gráfica ni conclusión en las slides.
- La simulación de la varianza de la AP (§20) es ilustrativa (scores binormales); la varianza real hay que medirla con bootstrap sobre nuestros datos.
- La sede del paper OOD CF (NLDL 2026) viene del borrador; el txt no la muestra en la cabecera.
