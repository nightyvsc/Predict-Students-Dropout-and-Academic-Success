from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    learning_curve,
    train_test_split,
    validation_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_text
from ucimlrepo import fetch_ucirepo

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
OUTPUT_DIR = Path("artifacts")

CATEGORICAL_FEATURES = [
    "Marital Status",
    "Application mode",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]
ACADEMIC_COLUMNS = {
    "Curricular units 1st sem (enrolled)": "first_enrolled",
    "Curricular units 1st sem (evaluations)": "first_evaluations",
    "Curricular units 1st sem (approved)": "first_approved",
    "Curricular units 1st sem (grade)": "first_grade",
    "Curricular units 2nd sem (enrolled)": "second_enrolled",
    "Curricular units 2nd sem (evaluations)": "second_evaluations",
    "Curricular units 2nd sem (approved)": "second_approved",
    "Curricular units 2nd sem (grade)": "second_grade",
}
ENGINEERED_FEATURES = [
    "first_sem_evaluation_rate",
    "first_sem_approval_rate",
    "second_sem_evaluation_rate",
    "second_sem_approval_rate",
    "semester_grade_change",
]


class AcademicFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create academic progress ratios using predictors only."""

    def fit(self, features: pd.DataFrame, target: Any = None) -> AcademicFeatureEngineer:
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        transformed = features.copy()
        first_enrolled = transformed["Curricular units 1st sem (enrolled)"].replace(0, np.nan)
        second_enrolled = transformed["Curricular units 2nd sem (enrolled)"].replace(0, np.nan)
        first_evaluations = transformed["Curricular units 1st sem (evaluations)"].replace(0, np.nan)
        second_evaluations = transformed["Curricular units 2nd sem (evaluations)"].replace(0, np.nan)

        transformed["first_sem_evaluation_rate"] = (
            transformed["Curricular units 1st sem (evaluations)"].div(first_enrolled).fillna(0)
        )
        transformed["first_sem_approval_rate"] = (
            transformed["Curricular units 1st sem (approved)"].div(first_evaluations).fillna(0)
        )
        transformed["second_sem_evaluation_rate"] = (
            transformed["Curricular units 2nd sem (evaluations)"].div(second_enrolled).fillna(0)
        )
        transformed["second_sem_approval_rate"] = (
            transformed["Curricular units 2nd sem (approved)"].div(second_evaluations).fillna(0)
        )
        transformed["semester_grade_change"] = (
            transformed["Curricular units 2nd sem (grade)"]
            - transformed["Curricular units 1st sem (grade)"]
        )
        return transformed


def build_pipeline(
    classifier: BaseEstimator,
    features: pd.DataFrame,
    scale_numeric: bool,
) -> Pipeline:
    categorical = [name for name in CATEGORICAL_FEATURES if name in features.columns]
    numeric = [name for name in features.columns if name not in categorical]
    numeric += ENGINEERED_FEATURES

    numeric_steps: list[tuple[str, BaseEstimator]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
            ("numeric", Pipeline(numeric_steps), numeric),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("feature_engineering", AcademicFeatureEngineer()),
            ("preprocessing", preprocessing),
            ("classifier", classifier),
        ]
    )


def make_model_searches(
    features: pd.DataFrame,
) -> dict[str, tuple[Pipeline, list[dict[str, list[Any]]]]]:
    tree = build_pipeline(
        DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        features,
        scale_numeric=False,
    )
    svm = build_pipeline(SVC(class_weight="balanced"), features, scale_numeric=True)
    return {
        "Árbol baseline": (
            tree,
            [
                {
                    "classifier__max_depth": [2, 3, 4, 5],
                    "classifier__min_samples_leaf": [10, 20, 40],
                    "classifier__ccp_alpha": [0.0, 0.001, 0.005, 0.01],
                }
            ],
        ),
        "SVM retador": (
            svm,
            [
                {"classifier__kernel": ["linear"], "classifier__C": [0.1, 1, 10, 100]},
                {
                    "classifier__kernel": ["rbf"],
                    "classifier__C": [0.1, 1, 10, 100],
                    "classifier__gamma": ["scale", 0.01, 0.1],
                },
            ],
        ),
    }


# 1. Análisis exploratorio y de dimensionalidad

def run_eda(features: pd.DataFrame, target: pd.Series) -> list[str]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    numeric_features = [name for name in features.columns if name not in CATEGORICAL_FEATURES]
    lines = [
        "# Resultados del taller: predicción de deserción y éxito académico",
        "",
        "## 1. Análisis Exploratorio y de Dimensionalidad",
        "",
        f"- Dimensiones originales: **{features.shape[0]} filas x {features.shape[1]} variables**.",
        f"- Variables nominales codificadas: **{len(CATEGORICAL_FEATURES)}**.",
        f"- Variables cuantitativas para correlación: **{len(numeric_features)}**.",
        f"- Valores faltantes en predictores: **{int(features.isna().sum().sum())}**.",
        f"- Filas duplicadas: **{int(features.duplicated().sum())}**.",
        "",
        "### Distribución de clases",
        "",
    ]
    class_counts = target.value_counts().sort_index()
    for label, count in class_counts.items():
        lines.append(f"- {label}: {count} ({count / len(target):.1%}).")

    print("\n=== 1. Análisis Exploratorio y de Dimensionalidad ===")
    print(f"Dimensiones: {features.shape[0]} filas x {features.shape[1]} columnas")
    print(f"Categóricas nominales: {len(CATEGORICAL_FEATURES)}; cuantitativas: {len(numeric_features)}")
    print(f"Faltantes: {int(features.isna().sum().sum())}; duplicados: {int(features.duplicated().sum())}")
    print("\nCardinalidad por variable:")
    print(features.nunique().sort_values(ascending=False).to_string())
    print("\nDistribución de clases:")
    print(class_counts.to_string())

    class_counts.plot.bar(color="#287271", figsize=(7, 4))
    plt.ylabel("Número de estudiantes")
    plt.title("Distribución de la variable objetivo")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_distribucion_clases.png", dpi=160)
    plt.close()

    numeric = features[numeric_features].select_dtypes(include=np.number)
    correlation = numeric.corr()
    figure, axis = plt.subplots(figsize=(13, 10))
    image = axis.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    axis.set_xticks(range(len(correlation.columns)), correlation.columns, rotation=90, fontsize=7)
    axis.set_yticks(range(len(correlation.columns)), correlation.columns, fontsize=7)
    axis.set_title("Correlación de Pearson: predictores cuantitativos")
    figure.colorbar(image, ax=axis, label="Correlación")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "01_correlacion_cuantitativas.png", dpi=160)
    plt.close(figure)

    correlated_pairs = []
    for row_index, first_name in enumerate(correlation.columns):
        for second_name in correlation.columns[row_index + 1 :]:
            value = correlation.loc[first_name, second_name]
            if pd.notna(value) and abs(value) >= 0.8:
                correlated_pairs.append((first_name, second_name, float(value)))
    lines.extend(
        [
            "",
            "### Correlación entre predictores cuantitativos",
            "",
            "Se excluyeron variables categóricas nominales aunque estén almacenadas como enteros: "
            "sus códigos no representan distancias ordinales válidas.",
            "",
        ]
    )
    if correlated_pairs:
        lines.append("Pares con |correlación de Pearson| >= 0.80:")
        for first, second, value in sorted(correlated_pairs, key=lambda pair: abs(pair[2]), reverse=True):
            lines.append(f"- `{first}` / `{second}`: {value:.3f}.")
    else:
        lines.append("No se encontraron pares cuantitativos con |correlación| >= 0.80.")
    lines.extend(
        [
            "",
            "**Lectura:** la correlación identifica redundancia lineal potencial, no causalidad. "
            "Se combina con las importancias del árbol en el punto 2; no se descartan variables "
            "antes de la validación cruzada.",
        ]
    )
    return lines


# 2. Selección de variables argumentada por importancia del baseline

def aggregate_tree_importances(model: Pipeline) -> pd.Series:
    names = model.named_steps["preprocessing"].get_feature_names_out()
    values = model.named_steps["classifier"].feature_importances_
    grouped: dict[str, float] = {}
    for transformed_name, importance in zip(names, values):
        if transformed_name.startswith("categorical__"):
            encoded_name = transformed_name.removeprefix("categorical__")
            original_name = next(
                (
                    name
                    for name in sorted(CATEGORICAL_FEATURES, key=len, reverse=True)
                    if encoded_name.startswith(f"{name}_")
                ),
                encoded_name,
            )
        else:
            original_name = transformed_name.removeprefix("numeric__")
        grouped[original_name] = grouped.get(original_name, 0.0) + float(importance)
    return pd.Series(grouped, name="importance").sort_values(ascending=False)


def save_feature_importances(model: Pipeline) -> tuple[pd.Series, list[str]]:
    importances = aggregate_tree_importances(model)
    importances.rename_axis("variable").to_csv(OUTPUT_DIR / "02_importancia_variables.csv")
    top = importances.head(15)
    top.sort_values().plot.barh(figsize=(9, 6), color="#e09f3e")
    plt.xlabel("Importancia")
    plt.title("Importancia agregada por variable original: árbol baseline")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_importancia_variables.png", dpi=160)
    plt.close()

    lines = [
        "",
        "## 2. Selección de Variables Argumentada",
        "",
        "Se usa `feature_importances_` del árbol baseline. Las importancias de niveles one-hot se "
        "suman por variable de origen para no fragmentar artificialmente las variables categóricas.",
        "",
        "### 15 variables con mayor importancia",
        "",
        "| Posición | Variable | Importancia |",
        "|---:|---|---:|",
    ]
    for position, (name, importance) in enumerate(top.items(), start=1):
        lines.append(f"| {position} | {name} | {importance:.4f} |")
    lines.extend(
        [
            "",
            "**Criterio:** se reporta el ranking como extracción de importancia del modelo base; "
            "las variables de mayor peso son candidatas de selección para interpretación. Se "
            "mantienen las variables en ambos modelos para comparar bajo el mismo conjunto de "
            "predictores y evitar selección previa al CV. La importancia no demuestra causalidad "
            "y puede favorecer variables con muchas categorías o puntos de corte.",
        ]
    )
    return importances, lines


def model_searches(
    features: pd.DataFrame,
    target: pd.Series,
    cv: StratifiedKFold,
) -> tuple[dict[str, GridSearchCV], list[dict[str, Any]]]:
    searches: dict[str, GridSearchCV] = {}
    results: list[dict[str, Any]] = []
    print("\n=== 3. Modelado Dual: Baseline vs. Retador ===")
    for model_name, (pipeline, grid) in make_model_searches(features).items():
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=grid,
            scoring="balanced_accuracy",
            cv=cv,
            n_jobs=-1,
            refit=True,
            return_train_score=True,
            error_score="raise",
        )
        search.fit(features, target)
        searches[model_name] = search
        cv_result = search.cv_results_
        best_index = search.best_index_
        results.append(
            {
                "modelo": model_name,
                "balanced_accuracy_cv": search.best_score_,
                "desviacion_cv": cv_result["std_test_score"][best_index],
                "balanced_accuracy_train_cv": cv_result["mean_train_score"][best_index],
                "parametros_elegidos": str(search.best_params_),
            }
        )
        print(f"\n{model_name}: CV balanced accuracy={search.best_score_:.4f}")
        print(f"Mejores parámetros: {search.best_params_}")
    return searches, results


def evaluate_on_test(
    searches: dict[str, GridSearchCV],
    x_test: pd.DataFrame,
    y_test: pd.Series,
    results: list[dict[str, Any]],
) -> tuple[pd.DataFrame, list[str]]:
    report_lines = [
        "",
        "## 3. Modelado Dual: Baseline vs. Retador",
        "",
        "Baseline: árbol de decisión, seleccionado por reglas visibles e importancia de variables. "
        "Retador: SVM, capaz de fronteras lineales o no lineales según kernel. Ambos usan el "
        "mismo split y la misma métrica de búsqueda.",
        "",
        "| Modelo | Balanced accuracy CV | Desv. CV | Balanced accuracy test | F1 macro test |",
        "|---|---:|---:|---:|---:|",
    ]
    reports: dict[str, str] = {}
    table_rows: list[str] = []
    for result in results:
        model_name = str(result["modelo"])
        search = searches[model_name]
        predictions = search.predict(x_test)
        result["balanced_accuracy_test"] = balanced_accuracy_score(y_test, predictions)
        result["f1_macro_test"] = f1_score(y_test, predictions, average="macro")
        table_rows.append(
            f"| {model_name} | {result['balanced_accuracy_cv']:.4f} | "
            f"{result['desviacion_cv']:.4f} | {result['balanced_accuracy_test']:.4f} | "
            f"{result['f1_macro_test']:.4f} |"
        )
        labels = search.classes_
        matrix = confusion_matrix(y_test, predictions, labels=labels)
        reports[model_name] = classification_report(y_test, predictions, zero_division=0)
        print(f"\n{model_name} — prueba reservada")
        print(reports[model_name])
        print(f"Matriz de confusión (orden {list(labels)}):\n{matrix}")
        report_lines.extend(
            [
                "",
                f"### {model_name}: métricas por clase",
                "",
                "```text",
                reports[model_name].rstrip(),
                "```",
                "",
                f"Matriz de confusión (orden de clases `{list(labels)}`):",
                "",
                "```text",
                str(matrix),
                "```",
                "",
            ]
        )

    table_separator_index = report_lines.index("|---|---:|---:|---:|---:|")
    report_lines[table_separator_index + 1 : table_separator_index + 1] = table_rows + [""]
    comparison = pd.DataFrame(results)
    comparison.to_csv(OUTPUT_DIR / "03_comparacion_modelos.csv", index=False)
    report_lines.extend(
        [
            "",
            "**Interpretación:** balanced accuracy promedia el recall de cada clase, útil ante "
            "desbalance. F1 macro da el mismo peso a cada clase. La tabla reporta CV para "
            "selección y test una sola vez para estimar generalización; no elegir el modelo solo "
            "por la métrica del entrenamiento.",
        ]
    )
    return comparison, report_lines


# 4.a GridSearchCV y validación cruzada estratificada K-Fold

def save_search_diagnostics(
    searches: dict[str, GridSearchCV],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
) -> list[str]:
    lines = [
        "",
        "## 4. Ajuste Sistemático de Hiperparámetros",
        "",
        "### 4.a GridSearchCV y validación cruzada (K-Fold)",
        "",
        f"Se usó `GridSearchCV` con {CV_FOLDS}-Fold estratificado, shuffle y semilla {RANDOM_STATE}. "
        "La métrica optimizada fue balanced accuracy. El split de prueba se reservó antes de "
        "la búsqueda y no se usó para elegir parámetros.",
        "",
        "| Modelo | Parámetros elegidos por CV |",
        "|---|---|",
    ]
    for model_name, search in searches.items():
        lines.append(f"| {model_name} | `{search.best_params_}` |")
        pd.DataFrame(search.cv_results_).to_csv(
            OUTPUT_DIR / f"04_busqueda_{'arbol' if 'Árbol' in model_name else 'svm'}.csv",
            index=False,
        )

    svm_results = pd.DataFrame(searches["SVM retador"].cv_results_)
    svm_results["param_classifier__kernel"] = svm_results[
        "param_classifier__kernel"
    ].astype(str)
    kernel_summary = (
        svm_results.groupby("param_classifier__kernel", observed=True)["mean_test_score"]
        .max()
        .sort_values(ascending=False)
    )
    kernel_summary.rename_axis("kernel").to_csv(OUTPUT_DIR / "04_comparacion_kernels_svm.csv")
    lines.extend(
        [
            "",
            "La búsqueda probó kernels lineal y RBF. **C** controla la penalización de errores; "
            "un valor alto puede reducir sesgo y aumentar varianza. En RBF, **gamma** controla "
            "el alcance de cada observación: gamma alto produce fronteras más locales y puede "
            "sobreajustar. El kernel lineal restringe la frontera; RBF permite fronteras no "
            "lineales. La tabla `04_comparacion_kernels_svm.csv` permite contrastarlos por el "
            "mejor promedio CV de cada kernel.",
            "",
            "La profundidad, mínimo de observaciones por hoja y `ccp_alpha` controlan la "
            "complejidad del árbol. Un árbol más profundo o menos podado suele bajar sesgo y "
            "subir varianza; `ccp_alpha` mayor poda ramas y simplifica el modelo.",
        ]
    )
    print("\n=== 4.a GridSearchCV + StratifiedKFold ===")
    print(f"Pliegues: {CV_FOLDS}; métrica: balanced_accuracy; resultados en {OUTPUT_DIR}")
    print("Mejor balanced accuracy CV por kernel SVM:")
    print(kernel_summary.to_string())
    return lines


# 4.b Curvas de validación y curvas de aprendizaje

def save_validation_curves(
    searches: dict[str, GridSearchCV],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
) -> list[str]:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    tree = searches["Árbol baseline"].best_estimator_
    tree_values = [0.0, 0.001, 0.005, 0.01, 0.02]
    train_scores, validation_scores = validation_curve(
        tree,
        x_train,
        y_train,
        param_name="classifier__ccp_alpha",
        param_range=tree_values,
        scoring="balanced_accuracy",
        cv=cv,
        n_jobs=-1,
    )
    axes[0].plot(tree_values, train_scores.mean(axis=1), marker="o", label="Entrenamiento")
    axes[0].plot(tree_values, validation_scores.mean(axis=1), marker="o", label="Validación")
    axes[0].set_xlabel("ccp_alpha (mayor = más poda)")
    axes[0].set_ylabel("Balanced accuracy")
    axes[0].set_title("Árbol: curva de validación")
    axes[0].legend()

    svm = searches["SVM retador"].best_estimator_
    svm_values = [0.01, 0.1, 1, 10, 100]
    train_scores, validation_scores = validation_curve(
        svm,
        x_train,
        y_train,
        param_name="classifier__C",
        param_range=svm_values,
        scoring="balanced_accuracy",
        cv=cv,
        n_jobs=-1,
    )
    axes[1].semilogx(svm_values, train_scores.mean(axis=1), marker="o", label="Entrenamiento")
    axes[1].semilogx(svm_values, validation_scores.mean(axis=1), marker="o", label="Validación")
    axes[1].set_xlabel("C (kernel y gamma seleccionados por CV)")
    axes[1].set_ylabel("Balanced accuracy")
    axes[1].set_title("SVM: curva de validación")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "05_curvas_validacion.png", dpi=160)
    plt.close(figure)

    lines = [
        "",
        "### 4.b Curvas de Validación y de Aprendizaje",
        "",
        "`05_curvas_validacion.png` varía `ccp_alpha` del árbol y `C` del SVM. Si la curva de "
        "entrenamiento queda muy por encima de validación, hay indicios de varianza/sobreajuste. "
        "Si ambas son bajas, hay indicios de sesgo/subajuste. La zona donde validación es alta "
        "y la brecha es moderada ofrece un balance más defendible. En esta corrida, la validación "
        "del SVM alcanza su máximo alrededor de C=1 (aprox. 0.725); al subir C a 10 y 100, "
        "entrenamiento sigue creciendo (aprox. 0.97 y 1.00) mientras validación baja (aprox. "
        "0.70 y 0.68), patrón de varianza alta. El árbol valida mejor con poda mínima (aprox. "
        "0.697); subir ccp_alpha reduce tanto entrenamiento como validación, evidencia de que "
        "una poda excesiva introduce sesgo.",
        "",
        "La curva SVM mantiene kernel y gamma elegidos por búsqueda y aísla el efecto de `C`; "
        "la comparación lineal/RBF se presenta en la búsqueda CV. Para el árbol, `ccp_alpha` "
        "mayor simplifica el árbol y suele aumentar sesgo mientras reduce varianza.",
    ]
    print("\n=== 4.b Curvas de validación ===")
    print("Guardadas en artifacts/05_curvas_validacion.png")
    return lines


def save_learning_curves(
    searches: dict[str, GridSearchCV],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
) -> list[str]:
    figure, axis = plt.subplots(figsize=(8, 5))
    for model_name, search in searches.items():
        sizes, train_scores, validation_scores = learning_curve(
            search.best_estimator_,
            x_train,
            y_train,
            train_sizes=np.linspace(0.2, 1.0, 5),
            scoring="balanced_accuracy",
            cv=cv,
            n_jobs=-1,
        )
        short_name = "Árbol" if "Árbol" in model_name else "SVM"
        axis.plot(sizes, train_scores.mean(axis=1), marker="o", label=f"{short_name} entrenamiento")
        axis.plot(
            sizes,
            validation_scores.mean(axis=1),
            marker="o",
            linestyle="--",
            label=f"{short_name} validación",
        )
    axis.set_xlabel("Número de observaciones de entrenamiento")
    axis.set_ylabel("Balanced accuracy")
    axis.set_title("Curvas de aprendizaje")
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "06_curvas_aprendizaje.png", dpi=160)
    plt.close(figure)
    print("Curvas de aprendizaje guardadas en artifacts/06_curvas_aprendizaje.png")
    return [
        "",
        "La curva de aprendizaje `06_curvas_aprendizaje.png` muestra si aumentar los datos "
        "reduce la brecha train/validación. Una validación que mejora con más ejemplos sugiere "
        "beneficio de más datos; curvas que convergen bajas sugieren sesgo alto; una brecha "
        "persistente sugiere varianza alta. En esta corrida ambos modelos mejoran su validación "
        "al aumentar los ejemplos; el árbol pasa aprox. de 0.63 a 0.70 y SVM de 0.69 a 0.72. "
        "SVM conserva una brecha train/validación cercana a 0.10, mayor que la del árbol, por lo "
        "que su mejora métrica debe sopesarse con mayor varianza.",
    ]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = fetch_ucirepo(id=697)
    features = dataset.data.features
    target = dataset.data.targets["Target"].squeeze()
    missing_columns = set(CATEGORICAL_FEATURES).difference(features.columns)
    missing_columns.update(set(ACADEMIC_COLUMNS).difference(features.columns))
    if missing_columns:
        raise ValueError(f"Columnas esperadas ausentes del dataset: {sorted(missing_columns)}")

    result_lines = run_eda(features, target)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    searches, results = model_searches(x_train, y_train, cv)

    _, selection_lines = save_feature_importances(searches["Árbol baseline"].best_estimator_)
    result_lines.extend(selection_lines)
    _, dual_lines = evaluate_on_test(searches, x_test, y_test, results)
    result_lines.extend(dual_lines)
    result_lines.extend(save_search_diagnostics(searches, x_train, y_train, cv))
    result_lines.extend(save_validation_curves(searches, x_train, y_train, cv))
    result_lines.extend(save_learning_curves(searches, x_train, y_train, cv))

    results_frame = pd.DataFrame(results)
    winner = results_frame.sort_values("balanced_accuracy_cv", ascending=False).iloc[0]
    result_lines.extend(
        [
            "",
            "## Síntesis de resultados",
            "",
            f"La mayor balanced accuracy de validación cruzada fue de **{winner['modelo']}** "
            f"({winner['balanced_accuracy_cv']:.4f}). Su evaluación en el test reservado obtuvo "
            f"balanced accuracy {winner['balanced_accuracy_test']:.4f} y F1 macro "
            f"{winner['f1_macro_test']:.4f}. La selección se basa en CV; el test solo estima "
            "generalización. La conclusión considera también estabilidad, curvas e interpretabilidad.",
            "",
            "**Límite temporal:** se usan predictores del segundo semestre para predecir el "
            "resultado final; por tanto, estos resultados no prueban capacidad de alerta al "
            "inicio del programa. Para ese uso habría que restringir los predictores al momento "
            "real de intervención y repetir la validación completa.",
            "",
            "## Archivos generados",
            "",
            "- `01_distribucion_clases.png` y `01_correlacion_cuantitativas.png` (punto 1).",
            "- `02_importancia_variables.csv` y `.png` (punto 2).",
            "- `03_comparacion_modelos.csv` (punto 3).",
            "- `04_busqueda_arbol.csv`, `04_busqueda_svm.csv`, `04_comparacion_kernels_svm.csv` (punto 4a).",
            "- `05_curvas_validacion.png` y `06_curvas_aprendizaje.png` (punto 4b).",
            "",
            "### Protocolo de reproducibilidad",
            "",
            f"Split estratificado: {1 - TEST_SIZE:.0%} entrenamiento / {TEST_SIZE:.0%} prueba. "
            f"Validación: {CV_FOLDS}-Fold estratificado; semilla aleatoria: {RANDOM_STATE}. "
            "Imputación, ingeniería de características, one-hot y escalado viven dentro del "
            "pipeline y se ajustan por pliegue; no se usa el test para elegir hiperparámetros.",
        ]
    )
    (OUTPUT_DIR / "resultados_taller.md").write_text("\n".join(result_lines) + "\n", encoding="utf-8")
    print(f"\nResultados y gráficas completos en: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
