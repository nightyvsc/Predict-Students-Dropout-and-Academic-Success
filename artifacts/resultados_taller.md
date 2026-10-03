# Resultados del taller: predicción de deserción y éxito académico

## 1. Análisis Exploratorio y de Dimensionalidad

- Dimensiones originales: **4424 filas x 36 variables**.
- Variables nominales codificadas: **17**.
- Variables cuantitativas para correlación: **19**.
- Valores faltantes en predictores: **0**.
- Filas duplicadas: **0**.

### Distribución de clases

- Dropout: 1421 (32.1%).
- Enrolled: 794 (17.9%).
- Graduate: 2209 (49.9%).

### Correlación entre predictores cuantitativos

Se excluyeron variables categóricas nominales aunque estén almacenadas como enteros: sus códigos no representan distancias ordinales válidas.

Pares con |correlación de Pearson| >= 0.80:
- `Curricular units 1st sem (credited)` / `Curricular units 2nd sem (credited)`: 0.945.
- `Curricular units 1st sem (enrolled)` / `Curricular units 2nd sem (enrolled)`: 0.943.
- `Curricular units 1st sem (approved)` / `Curricular units 2nd sem (approved)`: 0.904.
- `Curricular units 1st sem (grade)` / `Curricular units 2nd sem (grade)`: 0.837.

**Lectura:** la correlación identifica redundancia lineal potencial, no causalidad. Se combina con las importancias del árbol en el punto 2; no se descartan variables antes de la validación cruzada.

## 2. Selección de Variables Argumentada

Se usa `feature_importances_` del árbol baseline. Las importancias de niveles one-hot se suman por variable de origen para no fragmentar artificialmente las variables categóricas.

### 15 variables con mayor importancia

| Posición | Variable | Importancia |
|---:|---|---:|
| 1 | second_sem_approval_rate | 0.5557 |
| 2 | Curricular units 2nd sem (approved) | 0.1533 |
| 3 | Tuition fees up to date | 0.1025 |
| 4 | Course | 0.0564 |
| 5 | first_sem_approval_rate | 0.0408 |
| 6 | GDP | 0.0315 |
| 7 | second_sem_evaluation_rate | 0.0166 |
| 8 | Age at enrollment | 0.0141 |
| 9 | Admission grade | 0.0068 |
| 10 | Curricular units 2nd sem (grade) | 0.0068 |
| 11 | Curricular units 1st sem (enrolled) | 0.0064 |
| 12 | Curricular units 2nd sem (evaluations) | 0.0047 |
| 13 | Previous qualification (grade) | 0.0036 |
| 14 | Unemployment rate | 0.0008 |
| 15 | first_sem_evaluation_rate | 0.0001 |

**Criterio:** se reporta el ranking como extracción de importancia del modelo base; las variables de mayor peso son candidatas de selección para interpretación. Se mantienen las variables en ambos modelos para comparar bajo el mismo conjunto de predictores y evitar selección previa al CV. La importancia no demuestra causalidad y puede favorecer variables con muchas categorías o puntos de corte.

## 3. Modelado Dual: Baseline vs. Retador

Baseline: árbol de decisión, seleccionado por reglas visibles e importancia de variables. Retador: SVM, capaz de fronteras lineales o no lineales según kernel. Ambos usan el mismo split y la misma métrica de búsqueda.

| Modelo | Balanced accuracy CV | Desv. CV | Balanced accuracy test | F1 macro test |
|---|---:|---:|---:|---:|
| Árbol baseline | 0.6967 | 0.0159 | 0.6825 | 0.6683 |
| SVM retador | 0.7245 | 0.0158 | 0.7087 | 0.6963 |

### Árbol baseline: métricas por clase

```text
              precision    recall  f1-score   support

     Dropout       0.83      0.65      0.73       284
    Enrolled       0.38      0.64      0.48       159
    Graduate       0.84      0.75      0.80       442

    accuracy                           0.70       885
   macro avg       0.68      0.68      0.67       885
weighted avg       0.76      0.70      0.72       885
```

Matriz de confusión (orden de clases `['Dropout', 'Enrolled', 'Graduate']`):

```text
[[186  66  32]
 [ 28 102  29]
 [ 11  99 332]]
```

### SVM retador: métricas por clase

```text
              precision    recall  f1-score   support

     Dropout       0.82      0.68      0.74       284
    Enrolled       0.43      0.65      0.52       159
    Graduate       0.86      0.80      0.83       442

    accuracy                           0.73       885
   macro avg       0.70      0.71      0.70       885
weighted avg       0.77      0.73      0.74       885
```

Matriz de confusión (orden de clases `['Dropout', 'Enrolled', 'Graduate']`):

```text
[[193  62  29]
 [ 27 103  29]
 [ 16  73 353]]
```

**Interpretación:** balanced accuracy promedia el recall de cada clase, útil ante desbalance. F1 macro da el mismo peso a cada clase. La tabla reporta CV para selección y test una sola vez para estimar generalización; no elegir el modelo solo por la métrica del entrenamiento.

## 4. Ajuste Sistemático de Hiperparámetros

### 4.a GridSearchCV y validación cruzada (K-Fold)

Se usó `GridSearchCV` con 5-Fold estratificado, shuffle y semilla 42. La métrica optimizada fue balanced accuracy. El split de prueba se reservó antes de la búsqueda y no se usó para elegir parámetros.

| Modelo | Parámetros elegidos por CV |
|---|---|
| Árbol baseline | `{'classifier__ccp_alpha': 0.0, 'classifier__max_depth': 5, 'classifier__min_samples_leaf': 10}` |
| SVM retador | `{'classifier__C': 1, 'classifier__gamma': 'scale', 'classifier__kernel': 'rbf'}` |

La búsqueda probó kernels lineal y RBF. **C** controla la penalización de errores; un valor alto puede reducir sesgo y aumentar varianza. En RBF, **gamma** controla el alcance de cada observación: gamma alto produce fronteras más locales y puede sobreajustar. El kernel lineal restringe la frontera; RBF permite fronteras no lineales. La tabla `04_comparacion_kernels_svm.csv` permite contrastarlos por el mejor promedio CV de cada kernel.

La profundidad, mínimo de observaciones por hoja y `ccp_alpha` controlan la complejidad del árbol. Un árbol más profundo o menos podado suele bajar sesgo y subir varianza; `ccp_alpha` mayor poda ramas y simplifica el modelo.

### 4.b Curvas de Validación y de Aprendizaje

`05_curvas_validacion.png` varía `ccp_alpha` del árbol y `C` del SVM. Si la curva de entrenamiento queda muy por encima de validación, hay indicios de varianza/sobreajuste. Si ambas son bajas, hay indicios de sesgo/subajuste. La zona donde validación es alta y la brecha es moderada ofrece un balance más defendible.

En esta corrida, la validación del SVM alcanza su máximo cerca de C=1 (aprox. 0.725); al subir C a 10 y 100, entrenamiento sigue creciendo (aprox. 0.97 y 1.00) mientras validación baja (aprox. 0.70 y 0.68), patrón de varianza alta. El árbol valida mejor con poda mínima (aprox. 0.697); subir ccp_alpha reduce tanto entrenamiento como validación, evidencia de que una poda excesiva introduce sesgo.

La curva SVM mantiene kernel y gamma elegidos por búsqueda y aísla el efecto de `C`; la comparación lineal/RBF se presenta en la búsqueda CV. Para el árbol, `ccp_alpha` mayor simplifica el árbol y suele aumentar sesgo mientras reduce varianza.

La curva de aprendizaje `06_curvas_aprendizaje.png` muestra si aumentar los datos reduce la brecha train/validación. Una validación que mejora con más ejemplos sugiere beneficio de más datos; curvas que convergen bajas sugieren sesgo alto; una brecha persistente sugiere varianza alta.

En esta corrida ambos modelos mejoran su validación al aumentar los ejemplos; el árbol pasa aprox. de 0.63 a 0.70 y SVM de 0.69 a 0.72. SVM conserva una brecha train/validación cercana a 0.10, mayor que la del árbol, por lo que su mejora métrica debe sopesarse con mayor varianza.

## Síntesis de resultados

La mayor balanced accuracy de validación cruzada fue de **SVM retador** (0.7245). Su evaluación en el test reservado obtuvo balanced accuracy 0.7087 y F1 macro 0.6963. La selección se basa en CV; el test solo estima generalización. La conclusión considera también estabilidad, curvas e interpretabilidad.

**Límite temporal:** se usan predictores del segundo semestre para predecir el resultado final; por tanto, estos resultados no prueban capacidad de alerta al inicio del programa. Para ese uso habría que restringir los predictores al momento real de intervención y repetir la validación completa.

## Archivos generados

- `01_distribucion_clases.png` y `01_correlacion_cuantitativas.png` (punto 1).
- `02_importancia_variables.csv` y `.png` (punto 2).
- `03_comparacion_modelos.csv` (punto 3).
- `04_busqueda_arbol.csv`, `04_busqueda_svm.csv`, `04_comparacion_kernels_svm.csv` (punto 4a).
- `05_curvas_validacion.png` y `06_curvas_aprendizaje.png` (punto 4b).

### Protocolo de reproducibilidad

Split estratificado: 80% entrenamiento / 20% prueba. Validación: 5-Fold estratificado; semilla aleatoria: 42. Imputación, ingeniería de características, one-hot y escalado viven dentro del pipeline y se ajustan por pliegue; no se usa el test para elegir hiperparámetros.
