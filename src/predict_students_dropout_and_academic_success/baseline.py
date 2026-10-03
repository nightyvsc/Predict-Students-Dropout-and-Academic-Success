from ucimlrepo import fetch_ucirepo
from sklearn.compose import ColumnTransformer
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text


def main() -> None:
    dataset = fetch_ucirepo(id=697)
    features = dataset.data.features
    target = dataset.data.targets["Target"]

    categorical_features = [
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

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    preprocessing = ColumnTransformer(
        [("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    model = Pipeline(
        [
            ("preprocessing", preprocessing),
            (
                "classifier",
                DecisionTreeClassifier(
                    max_depth=4,
                    min_samples_leaf=20,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    print("Baseline: Decision Tree")
    print(f"Balanced accuracy: {balanced_accuracy_score(y_test, predictions):.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(model.classes_)
    print(confusion_matrix(y_test, predictions, labels=model.classes_))

    transformed_features = model.named_steps["preprocessing"].get_feature_names_out()
    importances = model.named_steps["classifier"].feature_importances_
    ranked_features = sorted(
        zip(transformed_features, importances), key=lambda item: item[1], reverse=True
    )
    print("\nTop feature importances:")
    for name, importance in ranked_features[:10]:
        print(f"{name}: {importance:.3f}")

    print("\nDecision tree:")
    print(
        export_text(
            model.named_steps["classifier"],
            feature_names=list(transformed_features),
        )
    )


if __name__ == "__main__":
    main()