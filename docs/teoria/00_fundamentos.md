# Fundamentos de la detección de anomalías

> **Fuentes:** slides_clasicas p01–p09 (Deteccion_de_anomalias.pdf, portada de V. Laparra; revisadas como imagen) · transcripcion.txt líneas 1–716 [00:00:05–01:13:45] y 1765–1947 [03:06:44–03:24:47] · tramos adicionales localizados con búsqueda en toda la transcripción: l. 935–959 [01:35:53–01:38:40], 1485–1595 [02:42:27–02:52:20], 2130–2213 [03:41:51–03:49:23], 2533–2549 [04:22:10–04:23:49] · papers/libro_IDA2017_cap_Hierarchical_Novelty_Detection.txt (Simeone, Santos-Rodríguez, McVicar, Lijffijt, De Bie; IDA 2017, pp. 310–321) · papers/Detecting_trends_in_twitter_time_series.txt (De Bie, Lijffijt, Mesnage, Santos-Rodríguez; IEEE MLSP 2016).
>
> **Convenciones:** [hh:mm:ss] = marca de la transcripción · "pN" = página de las diapositivas · "HND p. N" = página del libro IDA 2017 · "Twitter p. N" = página del paper MLSP 2016 · "(nota del redactor)" = contenido que no se dijo en clase.
>
> **Errores fonéticos corregidos de la transcripción:** "aulayer / una playa / off layer / Snoulier / ondayas / batalla" = outlier · "Dakes of Shift / chip / ir a selship / gaitas chiel" = data shift / covariate shift · "ahora distribution data / autopista / autogestitution data / auto distribution" = out-of-distribution · "menú par / nenupa" = nenúfar · "Call Start" = cold start · "rainforma learning" = reinforcement learning · "sale 500" = sal y pimienta · "Vino blanco para bebés" = ruido blanco para bebés · "This wise" = piecewise · "Smoke" = SMOTE · "redil" = Reddit · "quaris a de X" = consultas (queries) a X/Twitter.

---

## 0. Contexto de la sesión
**Idea.** La clase arranca recordando la sesión anterior de sistemas de recomendación: un problema de ML "normal" con circunstancias especiales (matriz usuario–objeto, *cold start*, *reinforcement learning* con feedback humano). La detección de anomalías se presenta como "un caso muy especial de machine learning" que, por ejemplo, habría servido para medir "cómo de nuevas son las recomendaciones" [00:04:24]–[00:05:49].

**Ponente.** La portada (p01) es de **Valero Laparra** (Departament d'Enginyeria Electrònica, Escola Tècnica Superior d'Enginyeria, valero.laparra@uv.es). Sin embargo, al cierre el ponente dice "a través de Valero me mandáis un correo" [04:22:10], así que **quien habla no es el autor de las slides**. Habla de "un ejemplo que hicimos" sobre los dos papers [00:19:15], [00:55:56], que comparten coautor (R. Santos-Rodríguez). (nota del redactor: es probable que el ponente sea un coautor de esos papers. Es una inferencia y no está confirmada. En este documento se le llama "el profesor".)

*Fuente: slide p01 · transcripción [00:04:24]–[00:05:49], [04:22:10].*

---

## 1. Nombres del problema y *one-class classification*
**Idea.** El mismo problema aparece con muchos nombres. El objetivo común es "identificar de forma automática que algo es diferente o raro" [00:06:06].

**Nombres que cita el profesor.**
- *Anomaly detection*, *novelty detection* y *fault detection* (transcrito "fair detection" / "fallo detection") [00:05:55], [00:09:58].
- *Outlier detection* ("que decíamos ayer") [00:06:15].
- *Out-of-distribution (OOD) data*: "lo veremos más adelante" [00:06:19] (ver §14).
- *One-class classification*: "ese es el nombre final que tiene" [00:07:12]–[00:07:14].
- Recapitula en [00:46:05]–[00:46:21] ("outlier detection, detección de novedad, detección de anomalía, detección de fallos, clasificación con una sola clase") y añade un problema muy relacionado, la **extracción de tendencias** (§12).
- En la slide p03 los términos aparecen juntos: **"Anomalia/Outlier/Ruido"**.

**One-class classification.** Frente al clasificador binario de la clase anterior (círculos frente a cruces), aquí solo hay una clase. Las soluciones típicas asumen que puede haber outliers "en cualquier sitio que no sea lo que caracteriza a esta clase", y hace falta **un umbral** que decida qué es outlier y qué no. El ejemplo es un "huevo" (región cerrada) que envuelve la clase: lo que no debería estar cae fuera [00:07:04]–[00:08:05]. Todas las técnicas conocidas siguen sirviendo [00:06:41]. "Casi cualquier clasificador de los que conocéis os valdrá", con pequeñas modificaciones, tanto si la novedad está en series temporales como en clasificación o regresión [01:01:40]–[01:02:03].

**Formalización en el capítulo HND.** *Novelty detection* consiste en decidir si los puntos nuevos pertenecen a alguna de las clases presentes en los datos. La forma natural es encerrar los datos en una superficie de decisión: lo que cae fuera es novedad. La superficie más común es la hiperesfera (SVDD, Tax & Duin) (HND p. 314):
- *Hard margin*: $\min_{R,\mu} R^2$ s.a. $R\ge 0$, $\|x_i-\mu\|^2 \le R^2$.
- *Soft margin*: permite que algunos puntos queden fuera, lo que lo hace más robusto entre train y test. $\min_{R,\mu,\xi} R^2 + C\sum_i \xi_i$ s.a. $\xi_i\ge0$, $\|x_i-\mu\|^2\le R^2+\xi_i$.
- *Multiclase*: una hiperesfera $(\mu_k,R_k)$ por clase, $\min \sum_k R_k^2 + C\sum_i\xi_i$ s.a. $\|x_i-\mu_{y_i}\|^2 \le R_{y_i}^2+\xi_i$ (HND p. 315).

*Fuente: transcripción [00:05:49]–[00:08:14], [00:09:54]–[00:10:06], [00:46:05]–[00:46:26], [01:01:05]–[01:02:03] · slide p03 · HND pp. 314–315.*

---

## 2. Definir qué es una anomalía es subjetivo
**Idea.** "El mayor problema que vamos a tener hoy es intentar definir qué es un outlier" [00:09:54]–[00:10:06]. "La definición de qué es y qué no es un outlier depende de nosotros. Y en función de lo que decidamos, las cosas tendrán una pinta muy diferente" [00:14:02]–[00:14:12]. "Eso es bastante raro en machine learning", porque hasta ahora no habíamos visto casos en los que algo sea o no outlier según cómo pensemos sobre ello [00:13:16]–[00:13:22].

**Ejemplos (verificados contra las slides).**
1. **Banco de peces (p02).** Peces azules de varios tonos y tamaños con un único pez rojo en el centro. "La mayoría diría que el outlier es este pez rojo… pero en realidad no hay ningún motivo para que yo diga eso." Los outliers podrían ser los peces pequeños, los grandes o los de otro tono, "o puede que todos pertenezcan a la misma clase y ninguno sea un outlier" [00:10:15]–[00:10:45].
2. **Nube verde y dos puntos rojos (p03, arriba izquierda).** Los dos puntos rojos están rodeados por un círculo con la etiqueta "OUTLIERS". Si nadie dice que son outliers, se puede ver como clasificación con dos clases. "Normalmente solemos entender, pero esto no tiene por qué ser verdad, que los outliers son menos frecuentes que la clase que nos interesa" [00:11:00]–[00:11:24].
3. **Tres clases y un punto rosa (p03, arriba centro).** Ejes X–Y con grupos rojo, morado y verde (la transcripción dice "roja, azul y verde"; en la slide el grupo es morado) y un punto rosa aislado con la etiqueta "Outlier". Aquí la definición no se basa en la densidad sino en "cómo de lejos esté del objeto de interés" [00:11:27]–[00:11:48].
4. **Las anomalías cambian con el tiempo.** Si siguen apareciendo 3 o 4 puntos rosas, "seguramente lo que es un outlier es otra clase" a la que habrá que poner nombre [00:11:48]–[00:12:14] (§4).
5. **Frecuencia de nombres y logaritmo (p03, abajo izquierda).** Gráfico de barras "Counts of Names" de Amelia a Sonny. Jane llega a ≈25, frente a ≤7 del resto, y está marcada como "outlier". Si se quita Jane, Jeremy (≈7) "parecerá un outlier; en realidad antes no lo era". Con el **logaritmo de la cuenta** "no va a aparecer ningún outlier": todas las columnas se parecen [00:12:23]–[00:13:16]. Moraleja: tanto la transformación de la variable como el conjunto de referencia cambian qué parece anómalo.
6. **Sueldo antes y después del máster (p03, abajo centro: tres mini-gráficos "Income at Start" frente a "Income at Finish").** Es un caso de regresión. Un punto parece lejos de los demás y "volvemos a tener el mismo problema: qué quiere decir lejos". "Si lo veo en rosa… no pertenece a la distribución. Pero si no lo coloreo, parece que hasta sigue el modelo que tengo" [00:13:28]–[00:14:01]. *Paneles corregidos tras ver la imagen:* (i) nube sin recta y punto rosa por encima; (ii) nube con recta de regresión y punto rosa muy arriba a la derecha, fuera de los ejes pero **sobre la prolongación de la recta** (es el que "sigue el modelo"); (iii) nube con recta y punto rosa arriba a la izquierda, **lejos de la recta**.
7. **Recuadro discontinuo (p03, derecha; no se comenta en clase).** Dos clusters azules (uno disperso arriba a la derecha, otro compacto abajo a la izquierda), un punto rojo pegado al cluster compacto y tres puntos grises aislados en el espacio vacío. (nota del redactor: ilustra la diferencia entre un outlier **local**, el rojo, que está cerca de un cluster denso pero fuera de él y es la idea de LOF, y outliers **globales**, los grises.)
8. **Ventana temporal (acción en bolsa).** Una caída vista con un mes de datos parece enorme. Con 10 años de la misma señal "la anomalía casi ni siquiera se ve". Con 100 años de bolsa aparecen los picos del 29, del 73 y del 91–92. Para inversiones a largo plazo, los picos del último mes "me van a dar completamente igual". "Depende de vosotros también decidir cuál es la ventana adecuada" [00:40:15]–[00:41:28].
9. **"Esto no es música, esto es ruido".** Decidir qué es ruido también es subjetivo [00:32:44]–[00:32:51] (§6).
10. **Clase mal caracterizada.** Con dos muestras no se puede saber si un punto lejano es outlier (§10) [01:02:46]–[01:04:39].

*Fuente: slides p02, p03 · transcripción [00:09:54]–[00:14:12], [00:40:15]–[00:41:28].*

---

## 3. Tipos de anomalía según dónde aparecen (local, global, dependiente, agrupada)
**Idea.** "Los outliers en general se pueden clasificar de varias maneras… Esta es una clasificación bastante visual pero útil" [00:14:21]–[00:14:30].

**Slide p04, verificada.** Ocho scatter plots 2D (2 filas × 4 columnas, con aspecto de proyecciones t-SNE) de puntos azules "Normal" y rojos "Anomaly". Cada columna lleva un título:
- **(a) Local.** Los rojos están dentro de los clusters densos, sobre todo en su periferia. "Si yo aquí no pintase los outliers de forma diferente, no sabría ni que son outliers… tienden a colocarse hacia las zonas exteriores… si todo fuera del mismo color, no veríamos ninguna diferencia" [00:14:35]–[00:15:10].
- **(b) Global.** Ya no se miran las estructuras individuales sino "la estructura general". Los rojos caen en los bordes y huecos de esa estructura global; en la figura aparecen dispersos entre clusters y alrededor de ellos [00:15:10]–[00:15:32].
- **(c) Dependency.** Hay dependencia entre la zona del espacio y la aparición de outliers. En la mayoría de zonas no hay ninguno y en zonas concretas sí: en la figura de arriba los rojos se concentran en la parte alta de la nube, y en la de abajo en ciertos clusters. Si uno de esos rojos fuera azul, "no tendría ni idea"; "si esto fuera no supervisado, parecería un punto normal" [00:15:32]–[00:16:08].
- **(d) Clustered.** Es una "situación bastante mejor": se conocen bien los datos y los outliers "tienden a concentrarse en zonas del espacio fáciles de identificar". En la figura, pequeños grupos rojos compactos y separados, arriba a la derecha en el panel superior y abajo a la derecha en el inferior. Corresponde al caso de **una clase nueva** que nunca se había visto [00:17:13]–[00:17:31].
- Recapitulación: "novedades locales, globales, que tienen dependencia con diferentes zonas del espacio, y estas tan bonitas en que aparecen clases nuevas" [00:30:06]–[00:30:15].

**Por qué importa (se retoma en la comparativa, fuera del tramo).** El rendimiento de los métodos "sí que va a ser muy diferente en función de si estamos hablando de locales, con dependencia, clustered o globales". En promedio sobre 57 datasets va "desde el 50 por 100 hasta casi el 100 por 100 en el mismo problema", y "esto es mucho más importante" que el grado de supervisión. "Si puedo saber qué tipo de problema tengo, bien, y si no, espera cualquier cosa entre casi aleatorio hasta que funcione perfecto" [03:46:34]–[03:47:11]. El consejo es: "si algo se puede visualizar, se visualiza. Intentamos entender en qué caso estamos" como primer paso [03:42:14]–[03:42:41]. (nota del redactor: la figura y los 57 datasets coinciden con **ADBench**, Han et al., NeurIPS 2022, cuya taxonomía define *local* (se desvía de su vecindario, como en LOF), *global* (lejos de todos los normales), *dependency* (rompe la estructura de dependencia entre variables de los normales) y *clustered/group* (anomalías parecidas entre sí que forman un grupo).)

**Etiquetar outliers es difícil.** Un humano reconoce un caballo en un dataset de gatos y perros, pero la probabilidad de que aparezca es tan baja que se pueden procesar datos "años y años sin haber visto ninguno". Cuando llegue, el detector tiene que estar preparado [00:16:14]–[00:17:06].

*Fuente: slide p04 · transcripción [00:14:18]–[00:17:31], [00:30:06]–[00:30:15], [03:42:14]–[03:47:11].*

---

## 4. Ciclo de vida de las clases: de anomalía a clase nueva
**Idea.** Una anomalía puede dejar de serlo sin que cambien los datos: cuando se acumula y alguien le pone nombre, pasa a ser una clase. "Suele haber un ciclo de vida de las categorías o las clases en cualquier problema de clasificación" [00:27:25].

- **Nenúfar.** Noticia "de ayer": una persona descubrió una especie nueva de nenúfar viendo una foto de Instagram, una muestra que cualquier otro habría tomado por un nenúfar normal. No es un outlier, "sino que es su propia clase". En cuanto se tiene más de una imagen y se le pone nombre, o se manda a un biólogo, "deja de ser un outlier, lo cual no tiene ningún sentido porque los puntos no han cambiado… ya no es una novedad ni una anomalía, semánticamente" [00:17:38]–[00:18:59]. "Desde el punto de vista de machine learning no vamos a hacer muchas cosas nuevas, pero desde el punto de vista de cómo llamamos a las cosas y cómo cambian en el tiempo, sí" [00:19:05].
- **Géneros musicales.** Una canción Z lejos del pop y del rock es un outlier, pero quizá sea el principio de la electrónica. Si se hace popular, Z deja de estar sola y aparece densidad a su alrededor. Un especialista dice "aquí ha pasado algo" y le pone nombre ("normalmente X es mi apellido porque lo he descubierto yo"; en música lo nombra la comunidad). Se convierte en un género establecido, tendrá subgéneros, envejecerá y quizá muera [00:26:00]–[00:27:20].
- **Extinción.** Si un animal se extingue, ya no hace falta categorizarlo [00:27:36]–[00:27:51].
- **Evolución** (Homo sapiens → Homo sapiens sapiens). Aparece una clase nueva **dentro** de la propia clase. El punto no está lejos de todo, sino cerca de las clases existentes pero "no pertenece del todo". Es un caso "mucho más interesante y muy difícil de detectar", y cuesta fijar el umbral para decir "a partir de aquí es una clase nueva" [00:27:51]–[00:28:59].

*Fuente: transcripción [00:11:48]–[00:12:14], [00:17:31]–[00:19:15], [00:26:00]–[00:29:37].*

---

## 5. Detección de novedad jerárquica (géneros musicales)
**Idea.** Es un ejemplo "que hicimos hace unos años": convertir la clasificación de géneros musicales en detección de novedad, buscando novedades **fuera** de todo lo existente y también **dentro** (subgéneros emergentes) para mandarlas a un especialista [00:19:15]–[00:29:29]. "Lo más importante es el ejemplo"; los artículos están por si alguien quiere verlos [00:29:49].

**En clase.**
- Cada persona tiene su propio árbol o grafo de géneros (Pop, Rock, Metal, Electrónica…; el punk rock dentro del rock, el blues rock dentro del blues). Existen estándares, y Google tiene un grafo aprendido a partir de datos para clasificar su contenido [00:19:54]–[00:20:53].
- Representación: una canción, o cualquier señal temporal, **"incluidos datos financieros"**, se pasa a **espectrograma** (tiempo × frecuencia, mediante una transformada de Fourier por ventanas), se trata como una imagen y se clasifica con una CNN [00:21:13]–[00:23:18].
- En la clasificación jerárquica las clases están relacionadas entre sí (géneros → subgéneros → …) [00:24:31]–[00:25:29]. En detección de novedad hay clases normales y muestras como Z lejos de todas [00:25:36]–[00:25:55]. "Dependerá de dónde coloque yo el umbral de lo que creo que es nuevo" [00:29:32].

**En el paper (HND, IDA 2017).**
- Cuatro problemas de la clasificación de géneros (HND p. 311): (1) jerarquía (si aplica Blues Rock, aplica Rock); (2) multietiqueta; (3) unas canciones solo encajan en géneros generales y otras son muy específicas; (4) las etiquetas no siempre son lo más específicas posible o faltan: "la presencia de una etiqueta implica que aplica, pero **la ausencia de una etiqueta no implica que no aplique**". Por eso, para cada clase solo hay realmente ejemplos positivos, lo que acerca el problema a *novelty detection*.
- Las relaciones entre géneros forman un árbol o un DAG (Blues Rock es hijo de Blues y de Rock), y "cualquier elección de estructura es solo una aproximación de la realidad" (HND pp. 312–313, Fig. 1).
- Enfoques jerárquicos: plano (solo las hojas), local (clasificadores encadenados de arriba abajo) y global o "big bang" (un único modelo) (HND p. 313).
- **Modelo HND**: una hiperesfera por nodo, anidadas, con la hija dentro de la madre (HND p. 316):
  $$\min_{R,\mu,\xi}\ \sum_{k=1}^K R_k + C\sum_{i=1}^n \xi_i \quad \text{s.a.}\quad \xi_i\ge0,\ R_k\ge0,\ \|x_i-\mu_{y_i}\|\le R_{y_i}+\xi_i,\ \|\mu_{Pa(k)}-\mu_k\|\le R_{Pa(k)}-R_k$$
  Es un problema convexo (SOCP) y siempre factible.
- **Interpretación de C** (HND pp. 316–317). Si $J$ es el número de puntos que quedan fuera de su hiperesfera en el óptimo, entonces $C \ge K/J$. Así, **C controla directamente cuántos outliers se toleran**: para tolerar exactamente uno, $C=K$. (nota del redactor: cumple un papel análogo a `nu` en `OneClassSVM` o a `contamination` en `IsolationForest`.)
- Fig. 2 (HND p. 315): (a) clasificación plana, (b) jerárquica, (c) novelty detection, (d) HND, donde un punto pertenece al nodo padre pero es novedad para sus subclases.
- Experimento (HND pp. 317–318): Million Song Dataset (features de Echo Nest) con tags de Last.fm, 13 géneros de 100 muestras cada uno y train/test al 50 %. Se dejaron 4 géneros fuera de la taxonomía (classical, country, folk, jazz) para que el método los descubriera como novedad. Los comparadores usan SVM lineal *one-vs-all*. Métricas: precisión y recall jerárquicos $hP=\frac{\sum_i|\hat P_i\cap\hat T_i|}{\sum_i|\hat P_i|}$ y $hR=\frac{\sum_i|\hat P_i\cap\hat T_i|}{\sum_i|\hat T_i|}$. La fracción de novedades se probó en el rango [0,005, 0,7].
- Resultados (HND pp. 319–320): HND supera al OVA plano, queda parecido a FLAT en cierto rango de C y por debajo de CLUS y HOVA. Subir C para descubrir más baja la precisión porque predice demasiadas etiquetas. Las taxonomías son arbitrarias, lo que debilita este tipo de enfoques. Trabajo futuro: elipsoides o una versión kernel.

*Fuente: transcripción [00:19:15]–[00:30:01] · HND pp. 310–321.*

---

## 6. Ruido
**Idea.** Es otra forma de abordar el problema. En sentido amplio, **el ruido es "algo que no es de interés"** [00:31:20]–[00:31:30]. La definición resulta confusa porque "hay gente que puede estar interesada en el ruido" [00:31:37]–[00:31:42]. En detección de anomalías se asume que los datos tienen **señal** (la estructura interesante que queremos aprender) y **ruido** (desorden irrelevante que normalmente queremos eliminar) [00:31:48]–[00:32:11]. "La detección de anomalías la podemos también definir como ser capaces de distinguir qué es ruido dentro de nuestros datos y qué no" [00:32:56]–[00:33:06].

**Slides.**
- p05: "Ruido · Definición", un gran signo "?" y una imagen de ruido blanco (estática gris de televisión).
- p06: "Definición: White Noise (plano) · Pink Noise (sonido) · **Distribución estadística!** · **Ruido =? error**". Tiene dos figuras. (1) "The Colors of Noise": densidad espectral de potencia (dB) frente a frecuencia (de $10^3$ a $10^5$ Hz, escala log). El violeta y el azul crecen con la frecuencia, el gris es aproximadamente plano (blanco), el rosa decrece suavemente y el marrón decrece más deprisa. (2) Dibujo a mano (dB frente a 20 Hz–1000 Hz–10000 Hz): White/Blanc plano, Pink/Rose decreciente, Brownian/Rouge con caída más pronunciada, Blue/Bleu y Violet crecientes, y Grey/Gris en forma de U.

**Ejemplos y explicación del profesor.**
- Imágenes: ruido gaussiano (algunos píxeles cambian de valor siguiendo una distribución) y ruido de sal y pimienta ("pequeños puntos bastante desagradables") [00:32:11]–[00:32:26]. Comunicaciones: una llamada que no se oye bien o unas obras de fondo [00:32:33]–[00:32:44].
- Demo de audio con ruido blanco, rosa ("más atenuado y suave") y marrón ("todavía más") [00:34:46]–[00:35:04]. Antes busca en vídeo "ruido blanco para bebés" de 12 h [00:33:55]–[00:34:05]. Para quien usa ese ruido para dormir o concentrarse, "el ruido no es ruido… es una señal que es útil" [00:35:16]–[00:35:30].
- El ruido blanco es el más habitual, sobre todo en comunicaciones, y "afecta a todas las frecuencias por igual" en el rango auditivo (20 Hz – unos kHz). **Ruido rosa (resuelto con p06):** la transcripción dice "va bajando sube amplitud según subimos en frecuencia", que es una mezcla. La slide aclara que **la amplitud o energía del rosa baja al subir la frecuencia**. Se puede diseñar cualquier ruido con cualquier función ("llamarlo con vuestro nombre") o pedirle a Claude un script que lo genere [00:35:39]–[00:36:51].
- Filtrado en telefonía: es "un filtrado de ruido que al final no hace más que ser una detección de novedad", con la diferencia de que aquí **se conoce la forma física del ruido y se sabe qué se busca**. En los géneros musicales, en cambio, "nadie nos ha dicho cuál era la definición de outlier" [00:36:51]–[00:37:28].
- Ruido como error: cualquier problema de ML se puede plantear como buscar "clasificadores que maximicen mi señal y minimicen mi ruido; minimizar mi ruido querrá decir minimizar mi error" [00:37:28]–[00:37:52].
- Segundo uso, en tendencias: el ruido "es cualquier cosa que no me interesa aprender, pero no estoy diciendo que todo el ruido sean outliers" [00:49:10]–[00:49:33] (§12).
- (nota del redactor: la densidad espectral es $S(f)\propto 1/f^\beta$, con $\beta=0$ para el blanco, 1 para el rosa, 2 para el marrón, −1 para el azul y −2 para el violeta. "Distribución estadística!" en p06 apunta a que el ruido se caracteriza por su distribución, lo mismo que hacen los métodos probabilísticos.)

*Fuente: slides p05, p06 · transcripción [00:30:41]–[00:38:00], [00:49:10]–[00:49:33].*

---

## 7. Enfoques según la información disponible: no supervisado, semisupervisado y supervisado
**Idea (slide p07: "ENFOQUES · No-supervisado · Semi-supervisado · Supervisado").** Es la "última forma de categorizar novedades": "por el proceso que seguimos a la hora de definir el problema" [00:38:17]–[00:38:30]. Resumen: hemos clasificado según **dónde** aparecen las novedades (§3) y según **qué información tengo**, es decir, si hay etiquetas o no [00:44:34]–[00:44:39].

**No supervisado: "el más clásico"** [00:41:36]. "Normalmente no suelo saber qué es una anomalía, con lo cual es difícil que alguien me la haya etiquetado" [00:38:42].
- Precio de una acción: se espera que suba, un evento (una guerra, una decisión de Trump) lo hunde y después se recupera. Quien mira la serie dice "esto ha sido un outlier" y lo correlaciona con las noticias [00:38:49]–[00:40:00].
- Una regla del tipo "si baja más de un 20 % en un día, tiene pinta de outlier"… "pero alguien me ha tenido que decir eso" [00:40:03]–[00:40:13].
- Influye la ventana temporal (§2, ej. 8).

**Supervisado: cuando "controláis los sistemas muy bien"** [00:41:43].
- **Tubería de petróleo en el desierto.** En lugar de mandar a una persona se pone un sensor. La serie es constante hasta que algo pasa, y el problema es decidir el **umbral** de lectura a partir del cual algo se ha estropeado [00:41:49]–[00:42:33]. Si se entiende bien la ingeniería de la tubería, "no necesito tener a alguien etiquetando todo el rato" [00:42:40]–[00:43:01]. En la práctica hay varios parámetros (presión, flujo) y no está claro cuándo todo va bien. Si con el conocimiento del sistema se pueden fijar las condiciones de outlier, "tenemos una definición clara de outlier… una situación más supervisada de forma fácil" [00:43:01]–[00:43:36]. (nota del redactor: este "supervisado" se apoya en una **definición experta** de la anomalía más que en un conjunto de etiquetas.)

**Semisupervisado: "normalmente estamos en una zona gris"** [00:43:40].
- Alguien etiquetó algunos picos de la acción, pero no todos, porque no tenía el conocimiento, no los vio o son de otra época. Se contrata a alguien que etiqueta durante un mes, y después siguen apareciendo picos que nadie etiquetará [00:43:42]–[00:44:07].
- Es habitual en datos financieros: primero etiqueta un experto, después alguien que no sabía nada y después "alguien que ha puesto a Claude a etiquetar… bien, mal o regular. Con eso tenéis que hacer todo lo que podáis" [00:44:10]–[00:44:28].
- En el mismo sentido (HND p. 311): la ausencia de etiqueta no implica que no aplique.

*Fuente: slide p07 · transcripción [00:38:17]–[00:44:39] · HND p. 311.*

---

## 8. Pocas etiquetas, desbalance y datos "ruidosos" (tramos fuera del rango)
**Las pocas anomalías etiquetadas, para validar y no para entrenar.** Un alumno pregunta si las anomalías conocidas sirven para fijar el umbral. El profesor responde que se podrían usar "como conjunto de validación… Si yo tuviera anomalías y son muy pocas, **nunca las utilizaría para entrenar el modelo, las utilizaría para validar las decisiones que tomaba con los umbrales**" [02:45:03]–[02:45:19]. "Si tenéis tantas anomalías como datos buenos, entonces sí las podéis utilizar para entrenar modelos… pero suele ser raro" [02:45:26]–[02:45:35]. El motivo es que las anomalías, "hasta que no tienen más densidad y se convierten en su propia clase", pueden aparecer en cualquier sitio del espacio [02:45:39]–[02:45:52]. Lo resume así: "si las tenéis, las podéis utilizar para lo que queráis: o bien para validar, o bien para entrenar" [02:52:06].

**Desbalance como caso límite.** La detección de anomalías puede verse como un multiclase extremo: una clase muy presente y muchas otras, desconocidas, con muy pocas muestras [02:47:16]–[02:47:39]. Nada impide entrenar un clasificador, pero se gastan recursos (complejidad) en una frontera mal definida [02:47:43]–[02:48:03]. Valen las soluciones típicas del desbalance: submuestrear la mayoritaria (que "tira cosas a la basura"), sobremuestrear con muestras sintéticas con ruido o usar **SMOTE**, que interpola entre muestras reales [02:48:07]–[02:50:04]. Advertencia: "asumid siempre que la distribución de los outliers va a ser una combinación de un millón de clases que nunca habéis visto… no tiene por qué ser que todos los outliers vengan de una nueva clase muy bonita" en una zona del espacio [02:48:55]–[02:49:10].

**Supervisión frente a tipo de anomalía (comparativa).** "Si yo tengo etiquetas y son muy pocas, lo más probable es que me dé bastante igual si es supervisado o no supervisado… las puedes utilizar para validación" [03:45:28]–[03:46:10]. El motivo es que "estamos intentando clasificar lo que tenemos contra el resto del mundo, y el resto del mundo no es fácil de caracterizar con 3 puntos" [03:45:51]–[03:45:55]. El tipo de anomalía pesa mucho más (§3).

**Datos ruidosos o corruptos.** La comparativa analiza qué pasa con **anomalías duplicadas**, **características irrelevantes** y **errores de anotación** [03:44:29]–[03:44:44]. Sobre los duplicados la transcripción no se entiende bien [03:47:57]–[03:48:07]. Con características irrelevantes "suele ser todo razonable" mientras su número sea comparable al de relevantes. Si se disparan (2 features útiles y 500 de ruido), "los métodos de distancia fallan mucho" y "detectan muchos outliers que no tienen nada que ver", sobre todo los basados en distancia y en probabilidad [03:48:07]–[03:49:18]. (nota del redactor: en ADBench, duplicar anomalías perjudica sobre todo a los métodos no supervisados, porque la anomalía deja de ser rara y densa, mientras que los supervisados aguantan mejor. Los errores de anotación perjudican más a los supervisados.)

**Elegir el enfoque.** Con regresión o reconstrucción se elige el modelo, con probabilidad la distribución, con distancias la métrica y con información las distribuciones. Si la elección es mala, el resultado es malo. "Siempre acaba dependiendo de vosotros decidir umbrales y qué aproximación creéis que es mejor" [02:50:11]–[02:51:09]. Muchos algoritmos clásicos "suelen asumir que yo no he visto nunca los outliers… o que es todo no supervisado" [02:51:49]–[02:52:00].

*Fuente: transcripción [02:44:51]–[02:52:06], [03:44:13]–[03:49:23].*

---

## 9. Familias de soluciones y librerías
- Slide p08 "SOLUCIONES": Distancias · Probabilidad · Información · Regresión/Reconstrucción · Kernels · Redes.
- En clase se formalizan "sobre todo en estas 3 categorías: distancias, probabilidad e información (teoría de la información)" [00:44:46]–[00:45:24]. Se empieza por las distancias en [01:02:03].
- Slide p09 "LIBRERIAS": Scikitlearn · pycaret · Pyod · "MAS: datos, libros, ejemplos...". Hay 2 o 3 scripts que exploran estas librerías. Con Claude, mirar el script "es bastante menos relevante". Casi cualquier clasificador o regresor sirve con pequeñas modificaciones [01:01:05]–[01:02:03].
- Evolución del campo (comparativa): primero aproximaciones estadísticas (gaussiana y desviaciones típicas), después métodos de distancia, después cualquier clasificador (árboles, One-Class SVM…) hasta ~2017, y redes después. "Desde 2022 casi todo lo llamamos *out-of-distribution* en vez de *outlier detection*" [03:43:02]–[03:43:48].

*Fuente: slides p08, p09 · transcripción [00:44:46]–[00:45:30], [01:01:05]–[01:02:03], [03:43:02]–[03:43:48].*

---

## 10. Distancia a la clase frente a distancia a la frontera
**Idea.** La regla general es "a más distancia, más anómalo", pero "no siempre es verdad" [01:02:15]–[01:02:22].

- **Pocas muestras.** Con una muestra por clase, las clases están muy mal caracterizadas (como el nenúfar de una sola foto). Un tercer punto lejano podría ser un outlier, o la clase real podría tener una forma que lo incluya. "Antes de empezar a pensar en distancias, tenéis que pensar si los datos que tenéis representan bien el concepto o categoría que queréis aprender. Si la respuesta es no, lo de la distancia es un poco irrelevante" [01:02:46]–[01:04:39]. Si la estructura se conoce bien (una pequeña esfera bien caracterizada), un punto fuera es claramente outlier aunque la distancia sea pequeña. La cantidad de datos necesaria depende de la complejidad de la estructura [01:04:44]–[01:05:19].
- **La misma distancia D en dos posiciones.** Un punto está entre las dos clases, sobre la frontera; otro está a la misma distancia pero al otro lado, lejos de ambas. El primero es "muy indeciso, pero creo que pertenece al problema"; el segundo está mucho más lejos de la otra clase y es "más claramente outlier". El profesor da por correcta la respuesta del alumno ("el de la derecha"), "pero depende de vuestras asunciones" [01:05:28]–[01:07:04].
- **Argumento contrario (SVM, MLP).** "A más distancia de la frontera, más seguro estoy de que pertenezco a la clase X". Una SVM asume que cuanto más lejos de la frontera, menos hay que preocuparse ("sé que es muy poco gato, así que tengo mucha confianza de que es un perro"). Los modelos discriminativos dan por bueno todo lo que queda a un lado, aunque no haya muestras, y "eso nos puede llevar a problemas": una muestra muy lejana puede ser de una clase nunca vista y el modelo la asignará con mucha confianza a "perro" [01:07:04]–[01:08:46].
- A continuación se plantea qué distancia usar (euclídea u otras) [01:13:31]–[01:13:59]; se trata en el tema de distancias.

*Fuente: transcripción [01:02:03]–[01:08:55], [01:13:31]–[01:13:59].*

---

## 11. Incertidumbre aleatoria y epistémica
**Idea.** Son dos tipos de incertidumbre, algo "importante para la vida en general" y ligado a distancias y outliers [01:09:04]–[01:09:32].

- Escenario: dos clases (círculos y X) separadas por una frontera de decisión [01:10:02]–[01:10:11].
- **Aleatoria (*aleatoric*).** Sobre la frontera, $P(\text{círculo}\mid x)=0{,}5$. Por muchas muestras que se añadan (el doble, diez más…), la duda en esa zona se mantiene. "Haga lo que haga, nunca voy a mejorar a la hora de decidir en esa zona": es **irreducible** [01:10:19]–[01:11:44].
- **Epistémica.** Es la incertidumbre en una zona lejana sin datos, y "sí puedo solucionarla teniendo más muestras" (10, 20 o 30 veces más), porque irán apareciendo muestras de baja probabilidad en zonas nuevas. Ejemplo: muestreando una gaussiana acaban saliendo valores extremos; entre millones de personas habrá alguien de 2,5 m. Con más muestras se **reduce** la incertidumbre hasta poder decir "esto pertenece a la clase cruz" [01:11:48]–[01:13:13].
- **Se retoma en probabilidad** [01:37:17]–[01:38:40]: "a menor probabilidad, más anómalo", "si nos acordamos de nuestro ejemplo anterior, aleatorio y epistémico". Un punto suelto, correctamente etiquetado pero lejos de la media, puede tener más probabilidad que otro que está "en medio de la nada" (entre clases). Los métodos probabilísticos dan resultados "ligeramente diferentes" a los de distancia.
- **Se retoma en regresión y reconstrucción** [02:43:34]–[02:44:45]. Tras proyectar los puntos sobre un modelo de regresión y medir el error con un umbral, aparecen dos casos: (a) la reconstrucción cae en una **zona densa** pero con un **error muy grande**, y el umbral lo marca como outlier; (b) el error respecto al modelo es **muy pequeño** pero **no hay datos en esa zona**. "Es el ejemplo de antes de incertidumbre epistémica y aleatoria… si los 2 fueran outliers, nadie tendría ningún problema con ello, pero depende de vuestra interpretación." (nota del redactor: (b) es el caso epistémico, un punto bien explicado por el modelo pero sin datos alrededor. Un detector basado solo en el error de reconstrucción no lo marcaría y uno de densidad sí.)
- (nota del redactor: los puntos con mucha incertidumbre epistémica son candidatos a outlier u OOD. Los de mucha incertidumbre aleatoria están "dentro del problema", en la frontera, y son errores inevitables del clasificador.)

*Fuente: transcripción [01:09:04]–[01:13:24], [01:37:17]–[01:38:40], [02:43:34]–[02:44:45].*

---

## 12. Tendencias en series temporales, ruido y picos
**Idea.** **Extraer tendencias** es un problema muy relacionado y frecuente en finanzas [00:46:21]–[00:46:26]. En él se mezclan tres cosas: tendencia, ruido y outliers [00:50:39]–[00:50:49].

**En clase.**
- La tendencia es una recta o curva que representa "hacia dónde va el mercado con esa acción" [00:46:30]–[00:46:57]. "Lo he simplificado hasta la muerte": a menudo hay que trocear la curva en varias tendencias, y pueden ser cuadráticas o polinomios de orden mayor [00:47:03]–[00:47:59]. Dada una serie ruidosa, se busca una simplificación de su comportamiento [00:47:57]–[00:48:11].
- La señal es la tendencia, lo que importa para comprar o vender. Por encima hay un "ruido" que da igual a largo plazo [00:48:31]–[00:49:02]. Pero un pico mucho más alto que los demás sí interesa marcarlo, por ejemplo para tomar una decisión de inversión en ese momento [00:49:20]–[00:49:55].
- Procedimiento: restar la tendencia deja el "ruido" (algo tipo sinusoide con un pico). Se modela ese ruido, y la diferencia entre el residuo y el modelo de ruido revela "mi outlier de verdad" [00:49:58]–[00:50:39].
- **Tuits por día** sobre un tema (la guerra de Irán): con un criterio como "más de 2000 tuits es outlier" se puede contar la historia de la guerra a partir de los días anómalos [00:50:49]–[00:51:46].
- **Artistas en Twitter.** Consultas a X para miles de artistas con dos objetivos: la tendencia (si se vuelven más populares) y los outliers. "No vamos a ser capaces de predecirlos, pero sí de anotarlos a posteriori" [00:51:50]–[00:52:46]. Un disco nuevo genera un pico que solo prevén el artista o la discográfica, igual que lo que hará Trump mañana o el que tiene información privilegiada ("me haré de oro y luego acabaré en la cárcel") [00:52:46]–[00:53:34]. A posteriori sí se puede ver que una farmacéutica sacó un medicamento y decir "aquí algo ha pasado" [00:53:34]–[00:53:57]. Lo interesante es cuando **el experto no encuentra explicación**: eso merece mirarse más. Ejemplo: GameStop, sin nada en los fundamentales ni en las noticias, con la gente organizada en Reddit [00:53:57]–[00:54:45]. Anotar sirve "para luego aprender para el futuro" [00:54:45]–[00:54:53].
- **Picos previsibles**: el anuncio de una canción de Aitana (Twitter o YouTube dimensionan la demanda), o Telefónica, que prevé las horas de levantarse y acostarse y los partidos de Champions [00:54:53]–[00:55:51].
- Método del paper ("tiene más de 10 años"): extraer la tendencia (curva roja **lineal a trozos**, aunque podría ser exponencial o cuadrática), restarla y detectar los picos [00:55:56]–[00:56:32]. Ampliación en [03:19:53]–[03:21:11]: tendencia, componente estacional (vacaciones, Navidad, "equivalentes a la gripe o las crisis"), varios tipos de ruido y picos. "Estamos descomponiendo una serie temporal en un montón de componentes" con un **modelo gráfico** en el que todas son variables aleatorias relacionadas.
- **Restricciones del dominio.** La serie de la acción tenía un pico grande positivo y otro negativo. En Twitter no puede haber picos negativos porque se cuentan tuits (0 o más). "Si tenéis propiedades de vuestros datos que os digan que algo no puede pasar… y lo metéis en vuestro modelo, conseguís modelos más robustos". Un modelo que permita picos negativos podría estimar valores por debajo de 0. "No vale la pena intentar detectar todo en todas partes" [00:56:32]–[00:58:17].
- **Anécdota de la startup.** Los artistas veían sus series y recibían una estimación de su probabilidad de éxito. El beneficio esperado era de unos 10.000 €/año y nadie dejó su trabajo. Había un **sesgo de selección**: quien paga esos servicios suele ser quien no va a triunfar ("Aitana nunca nos mandó una canción"), así que "acertábamos por definición". "Siempre se suele acertar mirando para atrás, pero nunca mirando para delante" [00:58:17]–[01:00:39].

**En el paper (MLSP 2016).**
- Motivación (Twitter p. 1): análisis técnico financiero, monitorización industrial (parques eólicos, reactores) y series de interés público.
- Una serie de Twitter se compone de (1) tendencia, (2) ráfagas concentradas y arbitrariamente grandes y (3) ruido. Supuestos (Twitter pp. 1–2): son **cuentas**, la mayoría está cerca de una tendencia lenta, las desviaciones positivas grandes son posibles pero **dispersas** y **no hay picos negativos**.
- Modelo (Twitter p. 2): $y_t\sim\text{Poisson}(x_t z_t)$, con pico multiplicativo $z_t\ge1$. En logaritmos, $y_t\sim\text{Poisson}(\exp(\chi_t+\zeta_t))$.
- *Prior* de la tendencia: $p(\chi)\propto\exp(-\lambda_1\|\nabla^2\chi\|_1)$, con $\nabla^2\chi_t=\chi_{t+1}-2\chi_t+\chi_{t-1}$ (relajación L1 del número de cambios de pendiente). Así $\chi$ es lineal a trozos y $x$ es **exponencial a trozos** (Twitter pp. 2–3). *Prior* de los picos: $z_t\sim$ Pareto (cola pesada), equivalente a $\zeta_t\sim\text{Exp}(\lambda_2)$, $\zeta_t\ge0$ (Twitter p. 3).
- MAP convexo, resoluble con CVX (Twitter p. 3): $\min_{\chi,\zeta}\ \lambda_1\|\nabla^2\chi\|_1+\sum_t\big(\lambda_2\zeta_t-(\chi_t+\zeta_t)y_t+e^{\chi_t+\zeta_t}\big)$ s.a. $\zeta\ge0$.
- **Teorema 2.2** (Twitter p. 4): si $y_t\le\lambda_2$, entonces $\zeta_t=0$, lo que da una forma sencilla de fijar $\lambda_2$. Límites: con $\lambda_1\to0$ no hay picos y con $\lambda_1\to\infty$ queda una única exponencial. Estacionalidad: $\exp(\chi_t+\zeta_t+\pi_{(t-1)\bmod k+1})$ con $\sum\pi_i=0$.
- Trabajo relacionado (Twitter pp. 4–5): Hodrick–Prescott (L2) y el filtro L1 con extensión de picos. Se descartan porque una tendencia lineal a trozos no tiene sentido en una serie positiva, el pico debería medirse **en relación con** la tendencia, el ruido gaussiano aproxima mal un Poisson de cuentas pequeñas y admiten picos negativos.
- Experimentos: datos sintéticos (10 series de 100 puntos con 3 picos), donde hay menos falsos positivos y negativos que con L1. Datos reales: 19.630 artistas de Reino Unido (reverbnation), 6.959.940 tuits entre dic-2014 y mar-2015, con anotación manual como *ground truth*. **$\lambda_2$ se fija en el percentil 80 de $y$, "como máximo un 20 % de picos"**. El método detecta la mayoría de picos anotados, encuentra eventos que el anotador pasó por alto y evita picos sin evento. Su ventaja es la **escalabilidad** (Twitter pp. 5–6). Conclusión: "un buen estimador de la tendencia depende del problema".

*Fuente: transcripción [00:45:30]–[01:00:39], [03:19:53]–[03:21:19] · Twitter pp. 1–6.*

---

## 13. Data shift (covariate shift y cambio en las etiquetas)
**Idea.** OOD es "el caso extremo" de otro problema, el **data shift**: cambios en los datos, en las etiquetas o en ambos. "Esto va a pasar mucho en la práctica" [03:09:57]–[03:10:18].

**Ejemplo del hospital.** Un clasificador decide si el paciente está listo para irse a casa en 24 h a partir de series de temperatura, ritmo cardiaco, respiración y saturación [03:10:26]–[03:11:36].
- Anomalías "normales": casos raros y errores de registro. Si un sensor de respiración se desconecta y marca 0 durante un día, no es que el paciente haya muerto: hay que mirar por qué [03:11:40]–[03:12:09].
- **Cambio en los datos (covariate shift).**
  - **COVID**: una enfermedad nueva que no estaba en la población original. Cambia la distribución de los datos y probablemente también la frontera de decisión, de forma relativamente suave [03:12:15]–[03:13:00].
  - **Pirámide poblacional**: un cambio aún más lento. Enfermedades antes poco comunes pasan a ser comunes; hay más puntos en ciertas clases, o el mismo número en zonas distintas del espacio. "Todo ese tipo de problemas entran dentro de la categoría de data shift / covariate shift" [03:13:00]–[03:13:37].
- **Cambio en las etiquetas.** El criterio de alta era variabilidad cardiaca del 10 %, saturación del 95 % y 36,5 °C. Nuevos estudios dicen que entre 36,1 y 36,9 está bien, y a partir de ahí se manda a casa a gente con 36,1. "El punto no ha cambiado, pero las etiquetas sí" [03:15:02]–[03:16:01]. "Podéis tener cambios en los datos, en las etiquetas, o de los 2" [03:16:05]–[03:16:12]. (nota del redactor: en la literatura, que cambie $P(y\mid x)$ con $P(x)$ fijo se llama *concept shift/drift*, y *label shift* en sentido estricto es que cambie $P(y)$ con $P(x\mid y)$ fijo. El profesor usa "cambio en las etiquetas" en sentido amplio, y su ejemplo es un *concept shift*.)

**Monitorización en producción.** "En la práctica pasa un montón cuando tenéis un sistema en producción, a los de finanzas también". Se vigila la salida de los modelos y se guarda **un conjunto de validación en producción** para estimar el error, porque habrá momentos en que el modelo empiece a fallar porque la distribución ha cambiado [03:13:54]–[03:14:23].

**Qué hacer.**
1. **Deshacer la transformación**: encontrar una transformación que devuelva los datos a donde estaban. El ejemplo dibujado es una traslación sencilla entre los datos de hace 5–6 años y los actuales: se estima la traslación y se corrigen los datos, y "los modelos siguen valiendo". Es un "caso ideal", pero "muchas veces con el tiempo somos capaces de entender en qué dirección se están moviendo las cosas" [03:14:31]–[03:14:41], [03:21:27]–[03:22:45].
2. **Reentrenar** aceptando que las cosas han cambiado [03:14:41].
3. **Adaptar los datos antiguos** si se sabe cómo. Por ejemplo, si el criterio de temperatura cambia para siempre, reetiquetar hacia atrás a los que tenían 36,4 y reentrenar. Si no se sabe, reentrenar e ir olvidando los datos antiguos [03:14:47]–[03:14:54], [03:17:13]–[03:17:42], [03:22:45]–[03:22:52].
4. **Marcar periodos.** En COVID los criterios cambiaron temporalmente (gestión del hospital, falta de camas) y luego se revirtieron. "A la hora de entrenar no entrenamos con esos datos ya nunca más": se **marcó en el tiempo** cuándo cambiaron y esos datos no se usan como normales. Ante una pregunta sobre crisis financieras, lo probable es "modelarlo aparte y separar los datos para no entrenar con ellos los modelos normales"; "puede que necesites modelos específicamente para eso" [03:17:42]–[03:18:28], [03:21:14]–[03:21:19].
- Fuera de los ámbitos financiero y médico, "lo que se suele hacer es ignorar que esto pasa, esperar que todo vaya bien y poner alertas de vez en cuando para reentrenar". "Era el caso más nuestro en [¿Fiverr?]": se asume que las poblaciones se mantienen constantes y se reentrena cuando hace falta, no a diario. En el hospital hay que hacerlo continuamente [03:16:14]–[03:17:02].
- Gripe frente a crisis: la gripe es estacional y previsible (picos entre octubre y febrero, "todo modelado"), las crisis no (teorías de "cada 20 años", el mérito de quien predijo 2007–08 y apostó su dinero). "Lo que se suele hacer es tener **modelos más a largo plazo y modelos más a corto plazo**" [03:18:28]–[03:19:43].

*Fuente: transcripción [03:09:57]–[03:19:43], [03:21:14]–[03:22:52].*

---

## 14. Out-of-distribution (OOD) como caso extremo, y explicabilidad
**Idea.** OOD "es el mismo problema que hemos visto hasta ahora y se resuelve con las mismas técnicas, pero es el nombre guay que le dieron hace 5 o 6 años". Si no se cambia el nombre, la gente se cansa. Con el nombre nuevo llegó la investigación y "aparecieron nuevas soluciones que nunca se habían utilizado" [03:07:41]–[03:08:50].

- La motivación es algo distinta: no se asume un comportamiento no deseado ni un fallo, sino "simplemente cosas que no conocemos, que están fuera de la distribución" (pocos datos de otros animales en un modelo de gatos y perros) [03:09:01]–[03:09:31].
- Se detectan porque "no queremos utilizarlos para entrenar ni para validar". Si aparecen caballos en un buen modelo de gatos y perros y se usan para entrenar, **"me van a cambiar la frontera"**, y en la evaluación meten ruido ajeno a la tarea [03:09:31]–[03:09:57].
- Es el caso extremo del data shift: los datos se han movido tanto "que ya no pertenecen a la misma distribución; por mucho que intentes corregir, lo que tienes es otra cosa". Se intenta detectar **el momento** del cambio para decir que "lo nuevo son todos outliers para el modelo anterior". El cambio puede ser continuo o drástico (una epidemia que lo cambia todo de un día para otro) [03:22:52]–[03:23:34]. "Tiene que ver con la **velocidad del cambio** y con mi **capacidad de adaptarme** al cambio" [03:23:40].
- **Explicabilidad.** Como no se asume que los outliers sean "malos" o fallos, se pueden hacer cosas más interesantes, como **explicar por qué** algo es outlier [03:23:47]–[03:24:27]. La explicación más sencilla es "tengo un umbral que son 5 y esto estaba a una distancia de 6". Después se busca "qué características dentro de esa muestra son las que han hecho que esta distancia sea esa" [03:24:37]–[03:24:47].
- Uso práctico que cuenta el profesor al final: en su trabajo y en "la primera startup" detectaban anomalías **sin supervisión** en registros o logs (antes con PCA, ahora con modelos tipo LLM), con un **sistema de alertas** para revisar después. La mayor parte del tiempo no había alertas y el problema era de infraestructura: almacenar los resultados [04:22:33]–[04:23:49].

*Fuente: transcripción [03:07:25]–[03:24:47], [03:43:44], [04:22:33]–[04:23:49].*

---

## Lo que dijo el profesor (recomendaciones explícitas)
- "Lo que queremos es identificar de forma automática que algo es diferente o raro." [00:06:06]
- "Casi cualquier clasificador de los que conocéis os valdrá… con pequeñas modificaciones todos deberían funcionar. Eso es lo único que quiero que recordéis." [01:01:52]–[01:02:03]
- Hace falta un umbral que diga qué es outlier y qué no, y "dependerá de dónde coloque yo el umbral". [00:07:51], [00:29:32]
- "La definición de qué es y qué no es un outlier depende de nosotros." [00:14:02]. Los outliers no tienen por qué ser menos frecuentes [00:11:14]–[00:11:24] y pueden cambiar con el tiempo [00:11:48].
- Transformar la variable (logaritmo) puede hacer desaparecer el outlier, y quitar uno puede crear otro (Jeremy). [00:12:51]–[00:13:16]
- Etiquetar outliers es muy difícil: el detector tiene que estar preparado para lo que no se ha visto en años. [00:16:14]–[00:17:06]
- El ruido es "algo que no es de interés", y detectar anomalías es distinguir qué es ruido y qué no. Pero "no todo el ruido son outliers". [00:31:30], [00:32:56], [00:49:14]–[00:49:20]
- "Depende de vosotros decidir cuál es la ventana [temporal] adecuada." [00:41:28]
- En finanzas es normal un etiquetado heterogéneo (experto, novato, Claude): "con eso tenéis que hacer todo lo que podáis". [00:44:10]–[00:44:28]
- Lo más interesante es el pico que el experto no sabe explicar. Anotar a posteriori sirve para aprender. [00:53:57]–[00:54:53]
- "Si tenéis propiedades de vuestros datos que os digan que algo no puede pasar… metedlo en vuestro modelo"; "no vale la pena intentar detectar todo en todas partes". [00:57:33]–[00:58:17]
- "Siempre se suele acertar mirando para atrás, pero nunca mirando para delante." [01:00:29]
- "No siempre es verdad que a más distancia, más anómalo"; antes de las distancias, pensad si los datos representan bien el concepto. [01:02:22], [01:04:29]
- Las SVM y los MLP asignan con mucha confianza puntos lejanos de la frontera aunque sean de una clase nunca vista. [01:08:28]–[01:08:46]
- La incertidumbre aleatoria no se reduce con más datos; la epistémica sí. [01:10:46]–[01:13:13]
- **Con muy pocas anomalías etiquetadas, no las uséis para entrenar sino para validar los umbrales.** Si hay tantas como normales, sí para entrenar. [02:45:03]–[02:45:35]
- Los outliers son "una combinación de un millón de clases que nunca habéis visto"; no esperéis que todos formen un grupo bonito. [02:48:59]–[02:49:10]
- Con pocas etiquetas da casi igual supervisado o no supervisado. Lo que más cambia el resultado es el **tipo de anomalía**. "Si algo se puede visualizar, se visualiza." [03:42:14], [03:46:00]–[03:47:11]
- Demasiadas características irrelevantes hunden a los métodos de distancia y de probabilidad. [03:48:12]–[03:49:18]
- Quitad lo OOD del entrenamiento y de la validación: mueve la frontera y mete ruido en la evaluación. [03:09:31]–[03:09:57]
- El data shift "va a pasar mucho en la práctica": guardad un conjunto de validación en producción y vigilad el error. [03:10:18], [03:14:04]–[03:14:23]
- Ante un shift: deshacer la transformación, reentrenar, adaptar los datos antiguos o marcar los periodos anómalos y no entrenar con ellos (o modelarlos aparte). Conviene tener modelos de largo y de corto plazo. [03:14:31]–[03:19:43], [03:21:19]
- OOD depende de la velocidad del cambio y de la capacidad de adaptarse. [03:23:40]
- Si se puede, explicad por qué algo es outlier (umbral frente a distancia y qué variables la causan). [03:24:21]–[03:24:47]

---

## Aplicación a nuestra práctica (fraude, AUPR, 0,17 %)
*(Toda esta sección es interpretación del redactor a partir de lo anterior. Son hipótesis que hay que comprobar en los notebooks, no resultados.)*

### A. Nuestro problema en la taxonomía del profesor

| Eje de la clase | Dónde cae el fraude | Consecuencia práctica | Notebook |
|---|---|---|---|
| **Definición (§2)** | Aquí la anomalía **no** es subjetiva: viene dada por `Class=1`. Pero "raro" no equivale a "fraude": hay transacciones legítimas raras (ruido de interés nulo) y fraudes que imitan lo normal. | Un detector de rareza (GMM, IsolationForest) mide *rareza*, no *fraude*. La AUPR penaliza cada transacción legítima rara que se cuele arriba del ranking. | 02 |
| **Tipo (§3, p04)** | Hipótesis: **mezcla**. Parte de los fraudes estarían en zonas de baja densidad (*global*) y otra parte concentrada en regiones concretas de algunas Xk (*dependency/clustered*), con un resto mezclado entre normales (*local*). El baseline (ROC 0,948 frente a AUPR 0,567) encaja con esto: casi todos los fraudes quedan por encima de la mayoría de normales, pero hay suficientes normales raros intercalados en la parte alta para bajar la precisión. | Según el profesor, el tipo pesa más que la supervisión. Proyectar (PCA/t-SNE/UMAP) y colorear por `Class`. Si es *clustered/dependency*, el supervisado gana. Si es *local*, añadir LOF o kNN. Si es *global*, GMM o IForest. | 01_EDA, 02 |
| **Supervisión (§7–8)** | **Semisupervisado con pocas etiquetas positivas**: 295 fraudes, de los que unos 59 caen en un test del 20 %. Son pocas en términos relativos, pero no "3 puntos". Puede haber ruido de etiqueta: un fraude no detectado figura como `Class=0` ("la ausencia de etiqueta no implica que no aplique", HND p. 311). | Comparar las dos estrategias que plantea el profesor: (i) **one-class** sobre normales con los fraudes **solo para validar** y elegir el modelo (el baseline del profesor); (ii) **supervisado** con validación cruzada estratificada. Con 295 positivos lo esperable es que (ii) o un híbrido ganen, pero hay que justificarlo con AUPR en CV, no darlo por hecho. | 02, 03, 04 |
| **Ciclo de vida (§4)** | Los defraudadores cambian de patrón. Un fraude nuevo empieza siendo outlier (epistémico, OOD) y solo se vuelve "clase" cuando se acumula y se etiqueta. | El supervisado solo reconoce los fraudes ya vistos. El componente de novedad cubre los que aún no se han visto. Este es el argumento del **híbrido**. | 04, 99 |
| **Distancia a frontera (§10)** | GBM, SVM o MLP darán "no fraude" con mucha confianza a un punto alejado de todo si cae del lado normal de la frontera. | Combinar el score supervisado con un score de densidad (−log p del GMM sobre normales) o tomar el máximo de los rangos de ambos. | 04 |
| **Incertidumbre (§11)** | Zonas donde se solapan fraude y no fraude: **aleatoria**, ni más datos ni más modelo lo arreglan. Zonas vacías: **epistémica**, que es donde aporta el detector de novedad. | Para la reflexión: explicar qué parte del error es irreducible y qué parte se cubre con el componente de novedad. | 99 |

### B. Riesgos de data shift de construcción a producción (§13–14)
- **Covariate shift ($P(x)$ cambia).** La producción (113.923 filas, sin etiqueta) puede venir de otro periodo, con otros hábitos de gasto, otros comercios u otra mezcla de `Amount`. Comprobaciones: KS o PSI por variable entre construcción y producción; **validación adversarial**, es decir, un clasificador que distinga construcción de producción (AUC ≈ 0,5 significa que no hay shift detectable); y comparar la distribución del score del GMM en ambos conjuntos.
- **Label shift ($P(y)$ cambia).** La prevalencia del fraude en producción puede no ser del 0,17 %. La AUPR depende de la prevalencia, pero el **ranking** no cambia, y el CSV solo pide scores ordenables. No calibrar umbrales con la prevalencia de construcción.
- **Concept shift ($P(y\mid x)$ cambia).** Aparecen nuevas tácticas de fraude. Es el riesgo principal para un modelo puramente supervisado, y por eso conviene el componente de novedad.
- **Qué hacer, siguiendo al profesor.** "Deshacer la transformación": si las variables solo se han desplazado o escalado, se puede estandarizar con estadísticas robustas (mediana/IQR) ajustadas en construcción y revisar si producción encaja. Evitar features frágiles al paso del tiempo. No reentrenar con producción usando etiquetas inventadas. Si se usa producción sin etiquetas (por ejemplo para ajustar el modelo de densidad de lo normal), justificarlo y tener en cuenta que contiene fraudes.

### C. El "ruido" de nuestros datos (§6, §8)
- **4.167 filas duplicadas.** (1) Hay **riesgo de fuga** si una copia cae en train y otra en test, lo que infla la AUPR de validación. Conviene deduplicar o agrupar los duplicados antes de partir (`GroupKFold` por hash de fila). (2) Si hay **fraudes duplicados**, para un detector no supervisado esos fraudes dejan de ser raros y forman pequeños grupos densos, que es el caso de "anomalías duplicadas" de la comparativa. Contar cuántos duplicados son `Class=1` y decidir con criterio explícito, documentado en el notebook. (3) Valorar si los duplicados son errores de registro (el "sensor a 0" del hospital) o transacciones legítimas repetidas.
- **`Amount`.** Tiene una cola muy pesada. Es el ejemplo de Jane: con cuentas brutas "todo lo grande" parece outlier, y con `log1p(Amount)` el panorama cambia. Probar ambas escalas y medir AUPR. Usar la restricción del dominio `Amount ≥ 0` (lección de los "picos no negativos"). Revisar las transacciones con `Amount = 0`.
- **X1..X20 anonimizadas, tipo PCA.** No se pueden explicar en términos de negocio, pero sí decir qué componentes empujan el score (la explicabilidad de §14). Si muchas variables son irrelevantes para el fraude, los métodos de distancia y densidad se degradan [03:48:12]: probar GMM y kNN sobre un subconjunto de variables seleccionado con información mutua o importancia del supervisado, validado siempre con AUPR.
- **OOD en entrenamiento** (§14). Normales extremadamente atípicos, como errores de registro, pueden "mover la frontera" del modelo de normalidad. Probar a recortar el 0,1–1 % más improbable de los normales antes de ajustar el GMM y medir si mejora la AUPR.

### D. Validación y entrega
- Los positivos son pocos: **CV estratificada repetida** (sin fuga por duplicados), informando de la media y la desviación de la AUPR y no de un único split. La AUPR solo depende del orden, así que no hace falta umbral para la métrica, aunque sí para explicar el modelo.
- Consejo del profesor sobre la validación en producción: dejar escrito en la reflexión cómo se monitorizaría el modelo (deriva del score y de las variables, alertas para reentrenar, modelos de corto y largo plazo).
- **Reflexión final (99_ENTREGA)**: anomalía subjetiva frente a etiqueta de fraude; tipo de anomalía observado en el EDA; por qué se usan (o no) las 295 etiquetas para entrenar y no solo para validar; incertidumbre aleatoria frente a epistémica; riesgos de shift y cómo se mitigan.

---

## Glosario ES / EN

| Español | English | Definición breve |
|---|---|---|
| Anomalía / valor atípico | Anomaly / outlier | Observación que se desvía de lo que se considera normal; la definición depende del problema (§2). |
| Detección de novedad | Novelty detection | Detectar en datos **nuevos** lo que no se parece al entrenamiento, que se asume limpio (one-class). |
| Detección de outliers | Outlier detection | Detectar atípicos **dentro** del propio conjunto de entrenamiento, que puede estar contaminado. |
| Detección de fallos | Fault / failure detection | Anomalía como mal funcionamiento de un sistema (tubería, sensor). |
| Clasificación de una clase | One-class classification | Aprender solo la clase normal y rechazar lo que cae fuera de su región (SVDD, One-Class SVM, GMM). |
| Fuera de distribución | Out-of-distribution (OOD) | Dato que no procede de la distribución de entrenamiento; nombre reciente del mismo problema (§14). |
| Dentro de distribución | In-distribution (ID) | Dato compatible con la distribución de entrenamiento. |
| Anomalía local / global | Local / global anomaly | Se desvía de su vecindario / de todos los datos (§3). |
| Anomalía dependiente | Dependency anomaly | Aparece en zonas concretas o rompe la dependencia entre variables (§3). |
| Anomalía agrupada / colectiva | Clustered / group (collective) anomaly | Anomalías parecidas entre sí que forman un grupo; posible clase nueva (§3). |
| Ruido | Noise | Variación que no es de interés; no todo el ruido es outlier (§6). |
| Ruido blanco / rosa / marrón | White / pink / brown (Brownian) noise | Espectro plano / que cae como $1/f$ / que cae como $1/f^2$ (p06). |
| Señal | Signal | Estructura de interés que se quiere aprender. |
| Umbral | Threshold | Valor del score a partir del cual se declara anomalía. |
| Puntuación de anomalía | Anomaly score | Valor continuo; cuanto más alto, más anómalo (nuestro CSV). |
| Contaminación | Contamination (`contamination`, `nu`) | Fracción esperada de anomalías; fija el umbral en IForest u OCSVM (≈ $C$ en HND). |
| No supervisado / semisupervisado / supervisado | Unsupervised / semi-supervised / supervised | Sin etiquetas / con pocas o parciales / con etiquetas completas (§7). |
| Desbalance de clases | Class imbalance | Una clase mucho más frecuente que otra (0,17 % de fraude). |
| Submuestreo / sobremuestreo / SMOTE | Undersampling / oversampling / SMOTE | Reequilibrar quitando mayoritarios / añadiendo minoritarios / interpolando minoritarios (§8). |
| Cambio en los datos | Data shift / dataset shift | La distribución conjunta cambia entre entrenamiento y uso. |
| Cambio de covariables | Covariate shift | Cambia $P(x)$ y se mantiene $P(y\mid x)$. |
| Cambio de etiquetas (prior) | Label (prior) shift | Cambia $P(y)$ y se mantiene $P(x\mid y)$. |
| Cambio de concepto / deriva | Concept shift / concept drift | Cambia $P(y\mid x)$: el mismo punto pasa a tener otra etiqueta (ejemplo de los 36,1 °C). |
| Validación adversarial | Adversarial validation | Clasificador entrenamiento/producción para detectar shift (nota del redactor). |
| Incertidumbre aleatoria | Aleatoric uncertainty | Irreducible, propia de los datos (solape en la frontera) (§11). |
| Incertidumbre epistémica | Epistemic uncertainty | Por falta de datos o conocimiento; se reduce con más datos (§11). |
| Tendencia / estacionalidad | Trend / seasonality | Componente lenta / componente periódica de una serie (§12). |
| Lineal a trozos | Piecewise linear | Tendencia formada por segmentos rectos (paper de Twitter). |
| Pico / ráfaga | Spike / burst | Desviación positiva, breve y grande de una serie de cuentas. |
| Error de reconstrucción | Reconstruction error | Distancia entre el dato y su reconstrucción (PCA, autoencoder, regresión). |
| Detección de novedad jerárquica | Hierarchical novelty detection (HND) | Hiperesferas anidadas por nodo de una taxonomía (§5). |
| Descripción de datos con vectores soporte | Support Vector Data Description (SVDD) | Hiperesfera mínima que encierra los datos (§1). |
| Precisión media / área bajo la curva PR | Average precision / AUPR | Métrica de ranking sensible a la prevalencia; la de la práctica. |
| Sesgo de selección | Selection bias | La muestra no representa a la población (anécdota de la startup). |
| Fuga de información | Data leakage | Información del test que llega al entrenamiento (duplicados entre folds). |
| Explicabilidad | Explainability / interpretability | Decir por qué algo es anómalo y qué variables lo causan (§14). |

---

## Cobertura de fuentes

| Fuente | Rango | Cubierto | Notas |
|---|---|---|---|
| slides p01 | portada | sí | §0. El ponente no es el autor de las slides ([04:22:10]). |
| slides p02 | banco de peces | sí | §2 ej. 1, verificado con la imagen |
| slides p03 | Anomalia/Outlier/Ruido (6 figuras) | sí | §1, §2 ej. 2–7. **Corregida** la descripción de los paneles de sueldo; el recuadro derecho se ha verificado y se describe con nota del redactor. |
| slides p04 | Local/Global/Dependency/Clustered (8 scatters) | sí | §3, verificado con la imagen; origen ADBench (nota del redactor) |
| slides p05 | Ruido · Definición · "?" · imagen de ruido blanco | sí | §6 |
| slides p06 | Colores del ruido · Distribución estadística · Ruido =? error | sí | §6. La contradicción sobre el ruido rosa se resuelve con la slide (decrece). |
| slides p07 | Enfoques | sí | §7 |
| slides p08 | Soluciones (6 familias) | sí | §9 (cada familia se desarrolla en otros temas) |
| slides p09 | Librerías | sí | §9 |
| transcripción l. 1–28 | [00:00:05–00:05:24] | sí | logística y repaso de recomendadores (§0) |
| transcripción l. 29–171 | [00:05:35–00:19:05] | sí | §1–§4 |
| transcripción l. 172–289 | [00:19:15–00:30:38] | sí | §4–§5 |
| transcripción l. 290–363 | [00:30:41–00:38:00] | sí | §6 |
| transcripción l. 364–434 | [00:38:17–00:45:24] | sí | §7, §9 |
| transcripción l. 435–585 | [00:45:30–01:00:39] | sí | §12 |
| transcripción l. 586–600 | [01:00:39–01:02:03] | sí | §9 |
| transcripción l. 601–716 | [01:02:15–01:13:41] | sí | §10, §11 |
| transcripción l. 935–959 (extra) | [01:35:53–01:38:40] | sí | incertidumbre en probabilidad (§11); los umbrales y la explicabilidad son de otro tema |
| transcripción l. 1485–1595 (extra) | [02:42:27–02:52:20] | sí | incertidumbre en reconstrucción (§11); pocas etiquetas para validar, desbalance, SMOTE (§8) |
| transcripción l. 1765–1771 | [03:06:44–03:07:16] | parcial | cierre del bloque anterior (intervalos de confianza), ajeno al tema |
| transcripción l. 1772–1947 | [03:07:25–03:24:47] | sí | §12–§14 |
| transcripción l. 2130–2213 (extra) | [03:41:51–03:49:23] | sí | visualizar primero, evolución del campo, supervisión frente a tipo de anomalía, duplicados e irrelevantes (§3, §8, §9). El detalle de la comparativa pertenece al tema de algoritmos y evaluación. |
| transcripción l. 2533–2549 (extra) | [04:22:10–04:23:49] | sí | identidad del ponente (§0); anomalías en logs (§14) |
| HND (libro IDA 2017) | pp. 310–321 | sí | §1, §5, §7. Lo que sigue al capítulo (otro capítulo, sobre asfalto y smartphone) queda excluido. |
| Twitter (MLSP 2016) | pp. 1–6 | sí | §12 |

**Cambios principales respecto al borrador:** se corrige la descripción de los tres paneles de sueldo de p03; se resuelve la contradicción del ruido rosa con p06; se distingue al ponente del autor de las slides; se integran las menciones a incertidumbre de [01:37:38] y [02:43:34]–[02:44:45]; se añade §8 (pocas etiquetas para validar, desbalance, SMOTE, supervisión frente a tipo, duplicados e irrelevantes), que es muy relevante para la práctica; se añaden la evolución del campo ("desde 2022, OOD"), la explicabilidad por variables y el caso de los logs; se corrigen errores fonéticos ("vino blanco" = ruido blanco, "Smoke" = SMOTE); se añaden el mapeo explícito a la taxonomía, los riesgos de shift, el ruido de nuestros datos y el glosario.

**Dudas pendientes**
- **"Fiverr" [03:16:35]** ("este era el caso más nuestro en Fiverr"). Por contexto es una empresa donde trabajó el ponente, con poblaciones estables y reentrenamiento ocasional, en contraste con el hospital. La transcripción no permite identificarla: queda como **dudoso**.
- Identidad del ponente: es una inferencia (coautor de los papers, posiblemente R. Santos-Rodríguez); no está confirmada.
- La frase sobre duplicados en la comparativa [03:47:57]–[03:48:07] no se entiende; el efecto de los duplicados se completa con nota del redactor (ADBench).
- "Hospital aparte de bellas" [03:15:10] puede ser el nombre de un hospital mal transcrito; no afecta al contenido.
