import unittest

import pandas as pd
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from predict_students_dropout_and_academic_success.pipeline import (
    ACADEMIC_COLUMNS,
    CATEGORICAL_FEATURES,
    AcademicFeatureEngineer,
    build_pipeline,
    make_model_searches,
)


class AcademicFeatureEngineerTests(unittest.TestCase):
    def make_features(self) -> pd.DataFrame:
        rows = []
        for index in range(12):
            row = {name: index % 3 for name in CATEGORICAL_FEATURES}
            row.update(
                {
                    "Curricular units 1st sem (enrolled)": 4,
                    "Curricular units 1st sem (evaluations)": 3,
                    "Curricular units 1st sem (approved)": 2,
                    "Curricular units 1st sem (grade)": 10 + index,
                    "Curricular units 2nd sem (enrolled)": 5,
                    "Curricular units 2nd sem (evaluations)": 4,
                    "Curricular units 2nd sem (approved)": 3,
                    "Curricular units 2nd sem (grade)": 11 + index,
                    "Age at enrollment": 18 + index,
                }
            )
            rows.append(row)
        return pd.DataFrame(rows)

    def test_engineers_ratios_and_grade_change(self) -> None:
        features = self.make_features()
        transformed = AcademicFeatureEngineer().fit_transform(features)

        self.assertAlmostEqual(transformed.loc[0, "first_sem_evaluation_rate"], 0.75)
        self.assertAlmostEqual(transformed.loc[0, "first_sem_approval_rate"], 2 / 3)
        self.assertAlmostEqual(transformed.loc[0, "second_sem_approval_rate"], 0.75)
        self.assertAlmostEqual(transformed.loc[0, "semester_grade_change"], 1.0)

    def test_both_allowed_model_pipelines_fit(self) -> None:
        features = self.make_features()
        target = pd.Series(["Dropout", "Enrolled", "Graduate"] * 4)

        tree = build_pipeline(
            DecisionTreeClassifier(max_depth=3, random_state=42),
            features,
            scale_numeric=False,
        )
        svm = build_pipeline(SVC(), features, scale_numeric=True)
        tree.fit(features, target)
        svm.fit(features, target)

        self.assertEqual(tree.predict(features).shape[0], len(features))
        self.assertEqual(svm.predict(features).shape[0], len(features))

    def test_searches_contain_only_tree_and_svm(self) -> None:
        searches = make_model_searches(self.make_features())

        self.assertEqual(set(searches), {"Árbol baseline", "SVM retador"})
        self.assertEqual(
            set(ACADEMIC_COLUMNS).difference(self.make_features().columns),
            set(),
        )
        self.assertIsInstance(searches["Árbol baseline"][0].named_steps["classifier"], DecisionTreeClassifier)
        self.assertIsInstance(searches["SVM retador"][0].named_steps["classifier"], SVC)


if __name__ == "__main__":
    unittest.main()
