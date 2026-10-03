# Predicción de deserción y éxito académico

Proyecto académico con el dataset UCI 697. El análisis y el pipeline están rotulados según los requisitos del taller; no se usan redes neuronales, ensambles, Naive Bayes ni PCA.

## Mapa del taller

| Requisito | Dónde se desarrolla | Evidencia principal |
|---|---|---|
| **1. Análisis Exploratorio y de Dimensionalidad** | `eda/eda.ipynb`; `pipeline.py` | Dimensiones, cardinalidad, faltantes, duplicados, balance, correlación cuantitativa y gráficas en `artifacts/`. |
| **2. Selección de Variables Argumentada** | `pipeline.py` | Importancias del árbol agregadas por variable original en CSV y gráfica; análisis de correlación en el notebook. |
| **3. Modelado Dual** | `pipeline.py` | Árbol baseline frente a SVM; métricas de CV y prueba, reporte por clase y matrices de confusión en `resultados_taller.md`. |
| **4.a GridSearchCV + K-Fold** | `pipeline.py` | Búsquedas de hiperparámetros acopladas a `StratifiedKFold`; mejores parámetros y tablas completas en `artifacts/`. |
| **4.b Curvas de Validación y Aprendizaje** | `pipeline.py` | `05_curvas_validacion.png`, `06_curvas_aprendizaje.png` e interpretación en el informe. |

## 1. Análisis Exploratorio y de Dimensionalidad

El notebook presenta tamaño, tipos, valores únicos por variable, descriptivos cuantitativos, valores faltantes, duplicados y distribución del objetivo `Target` (Dropout, Enrolled, Graduate). La matriz Pearson se calcula solo sobre columnas cuantitativas. Las variables categóricas están almacenadas como códigos numéricos, pero esos códigos no implican una distancia u orden; incluirlas en la matriz puede producir una lectura falsa.

Se reportan los pares de predictores cuantitativos con correlación absoluta de al menos 0.80 como indicio de redundancia lineal, no como causalidad ni regla automática para eliminar columnas. El desbalance de clases se considera al usar balanced accuracy y pesos de clase.

## 2. Selección de Variables Argumentada

La selección se argumenta con dos evidencias complementarias:

- **Filtro estadístico:** correlaciones Pearson entre variables cuantitativas, excluyendo variables nominales codificadas. Permite señalar pares redundantes; no mide relación causal ni cubre relaciones no lineales.
- **Importancia del baseline:** `feature_importances_` del árbol. Las importancias de categorías one-hot se suman de nuevo bajo la variable original para producir un ranking legible.

El ranking de importancia es evidencia predictiva dependiente del modelo, no una explicación causal; puede favorecer variables con más categorías o puntos de corte. Se conserva el conjunto de predictores en ambos modelos para compararlos bajo las mismas entradas y evitar una selección supervisada previa a la validación cruzada.

## 3. Modelado Dual: Baseline vs. Retador

**Baseline: árbol de decisión.** Se elige por su interpretabilidad: el informe imprime reglas, muestra la importancia agregada y acota complejidad con profundidad, tamaño mínimo de hoja y poda `ccp_alpha`. Sus hiperparámetros se buscan por CV; no están fijados a mano como resultado final.

**Retador: SVM.** Busca una frontera de separación con margen máximo. La búsqueda evalúa kernel lineal y RBF; el lineal restringe la frontera y el RBF puede capturar relaciones no lineales. El kernel ganador se decide mediante el promedio de validación cruzada, no se presupone. Las variables cuantitativas se escalan, paso especialmente importante para SVM.

Ambos modelos usan el mismo split estratificado, balanced accuracy como métrica de selección y `class_weight="balanced"`. Se informa balanced accuracy y F1 macro, reporte por clase y matriz de confusión. El resultado del test se presenta después de seleccionar por CV y no se usa para elegir parámetros o ganador.

La ingeniería de características agrega proporciones de unidades evaluadas sobre matriculadas, aprobadas sobre evaluadas en cada semestre y cambio de nota entre semestres. Son transformaciones de predictores, sin utilizar `Target`. Imputación, ingeniería, one-hot y escalado permanecen dentro de `Pipeline`, por lo que se ajustan dentro de cada pliegue sin fuga entre entrenamiento y validación.

**Límite temporal:** el modelo usa resultados del segundo semestre y predice la etiqueta final del dataset. Por tanto, estos resultados no demuestran que el modelo pueda alertar sobre deserción al inicio del programa. Para una intervención temprana habría que limitar predictores a los disponibles en el momento de decisión y volver a ejecutar la validación completa.

## 4. Ajuste Sistemático de Hiperparámetros

Los hiperparámetros son decisiones de configuración previas al entrenamiento; la búsqueda permite comparar configuraciones con el mismo criterio, en vez de elegirlas por ensayo y error. El objetivo es encontrar un nivel de complejidad que generalice y limitar el sobreajuste.

### 4.a GridSearchCV y validación cruzada K-Fold

`GridSearchCV` evalúa cada combinación declarada. Se usa `StratifiedKFold` con cinco pliegues: cada pliegue valida una ronda y los otros cuatro entrenan, conservando aproximadamente la proporción de clases. El criterio de búsqueda es balanced accuracy. Se informa su promedio y desviación entre pliegues para valorar rendimiento y estabilidad.

La búsqueda se ajusta solo con el 80% de entrenamiento. El 20% de prueba queda reservado y se evalúa una sola vez al final. La semilla y la estratificación hacen reproducible la división.

**Resultado de la corrida UCI:** el árbol obtuvo balanced accuracy CV `0.6967` y prueba `0.6825`; el SVM-RBF obtuvo CV `0.7245` y prueba `0.7087` (F1 macro `0.6963`). La mejora del SVM es moderada, no definitiva. RBF superó al kernel lineal por solo `0.0026` de balanced accuracy CV. `Enrolled` es la clase más difícil; SVM tuvo recall `0.65` y F1 `0.52` para ella.

**Parámetros explorados:**

- Árbol: `max_depth`, `min_samples_leaf` y `ccp_alpha`. Mayor profundidad, hojas pequeñas y menor poda permiten más complejidad; `ccp_alpha` más alto poda más ramas.
- SVM: `C` y kernel (`linear`, `rbf`); para RBF también `gamma`. `C` alto penaliza errores de entrenamiento más fuertemente y puede ajustar una frontera más compleja. `gamma` alto hace que cada observación influya en una zona más local; puede crear fronteras muy irregulares. Los resultados por kernel quedan en `04_comparacion_kernels_svm.csv`.

### 4.b Curvas de Validación y Curvas de Aprendizaje

`05_curvas_validacion.png` muestra entrenamiento y validación al cambiar `ccp_alpha` del árbol y `C` del SVM (con otros parámetros seleccionados por CV constantes). En general, complejidad excesiva puede dar puntuación de entrenamiento alta y validación inferior, señal de **varianza/sobreajuste**. Si ambas puntuaciones son bajas, hay **sesgo/subajuste**. Aumentar poda simplifica el árbol; un C menor limita el ajuste de SVM. Ambos cambios pueden reducir varianza, pero si se exageran aumentan sesgo.

`06_curvas_aprendizaje.png` muestra las puntuaciones de entrenamiento y validación a medida que crece la cantidad de datos. Una brecha persistente sugiere varianza; curvas que convergen con puntuaciones bajas sugieren sesgo. La curva también permite ver si agregar ejemplos parece mejorar la validación.

**Lectura de esta corrida:** en SVM, la validación alcanza aproximadamente `0.725` en `C=1`; con `C=10–100`, entrenamiento sube a `0.97–1.00` mientras validación cae a `0.70–0.68`, señal clara de varianza al aumentar la complejidad. El árbol valida mejor con poda mínima (`ccp_alpha` cercano a cero, balanced accuracy `~0.697`); podar más reduce tanto entrenamiento como validación, lo que indica sesgo por simplificación excesiva. En las curvas de aprendizaje, ambos mejoran al aumentar los datos: el árbol de `~0.63` a `~0.70` y SVM de `~0.69` a `~0.72`. SVM conserva una brecha train/validación cercana a `0.10`, mayor que la del árbol; su pequeña ventaja de test debe sopesarse frente a esa mayor varianza.

La comparación de kernel es categórica, por eso se interpreta desde los promedios de `GridSearchCV` para lineal y RBF, en vez de tratar el kernel como una escala numérica de una curva.

## Ejecución

Desde la raíz del repositorio, instala las dependencias y ejecuta el pipeline:

```bash
uv sync
uv run predict-students-dropout-and-academic-success
```

Se descarga el dataset desde UCI, por lo que hace falta conexión a internet. La búsqueda y las curvas implican múltiples ajustes y pueden tardar varios minutos. El notebook tiene un orden independiente para EDA; después de correr el pipeline, su celda final carga los artefactos existentes.

## Resultados generados

La ejecución crea `artifacts/` con:

- **Punto 1:** `01_distribucion_clases.png`, `01_correlacion_cuantitativas.png`.
- **Punto 2:** `02_importancia_variables.csv`, `02_importancia_variables.png`.
- **Punto 3:** `03_comparacion_modelos.csv` y métricas por clase/matrices en `resultados_taller.md`.
- **Punto 4.a:** `04_busqueda_arbol.csv`, `04_busqueda_svm.csv`, `04_comparacion_kernels_svm.csv`.
- **Punto 4.b:** `05_curvas_validacion.png`, `06_curvas_aprendizaje.png`.
- **Resumen de hallazgos:** `resultados_taller.md`, que incorpora métricas calculadas en esa ejecución y explica cómo leerlas.
