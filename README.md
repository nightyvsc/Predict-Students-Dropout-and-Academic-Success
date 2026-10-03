# Predict-Students-Dropout-and-Academic-Success

Pipeline de clasificación para analizar la deserción y el éxito académico de estudiantes universitarios usando el dataset 697 de UCI.

## Modelo baseline

El baseline usa un árbol de decisión poco profundo para mantener las predicciones interpretables. Las variables categóricas nominales se codifican con one-hot encoding para no tratarlas como cantidades ordenadas. El script separa los datos en entrenamiento y prueba (80/20), mantiene la proporción de clases y reporta balanced accuracy, precision, recall, F1, matriz de confusión e importancia de variables. También imprime las reglas del árbol.

La profundidad y el tamaño mínimo de hoja están fijados para limitar la complejidad del baseline; no se optimizan aquí. La búsqueda de hiperparámetros y las curvas de validación y aprendizaje corresponden a la comparación posterior con el modelo retador del proyecto.

## Ejecución

Instala las dependencias y ejecuta el baseline desde la raíz del proyecto:

```bash
uv sync
uv run predict-students-dropout-and-academic-success
```

El script descarga el dataset desde UCI, por lo que necesita conexión a internet.
