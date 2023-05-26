import json
import math
import pickle
import sqlite3
import sys
import textwrap
import typing as t
import numpy as np
import pandas as pd
import sklearn

from pathlib import Path
from sklearn import (
    datasets,
    dummy,
    ensemble,
    linear_model,
    metrics,
    model_selection,
    neighbors,
    neural_network,
    pipeline,
    preprocessing,
    svm,
    tree,
)


class SQML:
    conn: sqlite3.Connection

    def setup_schema(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self.conn.row_factory = sqlite3.Row
        schema_sql = (Path(__file__).parent / "sql" / "schema.sql").read_text()
        self.conn.executescript(schema_sql)

    def register_functions(self, conn: sqlite3.Connection) -> None:
        conn.create_function(
            "sqml_python_version", 0, self.python_version, deterministic=True
        )
        conn.create_function(
            "sqml_sklearn_version", 0, self.sklearn_version, deterministic=True
        )
        conn.create_function("sqml_load_dataset", -1, self.load_dataset)
        conn.create_function("sqml_train", -1, self.train)
        conn.create_function("sqml_predict", 2, self.predict)
        conn.create_function("sqml_predict_batch", 2, self.predict_batch)

    def python_version(self) -> str:
        return sys.version

    def sklearn_version(self) -> str:
        return sklearn.__version__

    def load_dataset(self, name: str, table: t.Optional[str] = None) -> str:
        mapping = {
            "iris": datasets.load_iris,
            "digits": datasets.load_digits,
            "wine": datasets.load_wine,
            "breast_cancer": datasets.load_breast_cancer,
            "diabetes": datasets.load_diabetes,
        }
        loader = mapping.get(name)
        table = table or f"dataset_{name}"

        if loader is None:
            return json.dumps(
                {
                    "error": f"Unknown dataset '{name}'. Available datasets: {', '.join(mapping.keys())}"
                }
            )
        else:
            bunch = loader()
            with self.conn:
                newline = ",\n"

                self.conn.execute(f"DROP TABLE IF EXISTS {table};")

                query = textwrap.dedent(
                    f"""
                    CREATE TABLE IF NOT EXISTS {table} (
                        {newline.join(f'[{name}] REAL' for name in bunch["feature_names"])},
                        target REAL
                    );
                    """
                )
                self.conn.execute(query)

                query = textwrap.dedent(
                    f"""
                    INSERT INTO {table} VALUES (
                        {', '.join(['?'] * (len(bunch["feature_names"]) + 1)) }
                    );
                    """
                )
                self.conn.executemany(
                    query,
                    np.hstack((bunch["data"], np.reshape(bunch["target"], (-1, 1)))),
                )

            task_type = "classification" if "target_names" in bunch else "regression"
            return json.dumps(
                {
                    "table": table,
                    "feature_names": list(bunch["feature_names"]),
                    "target_names": bunch["target_names"].tolist()
                    if task_type == "classification"
                    else [],
                    "size": len(bunch["data"]),
                }
            )

    def train(
        self,
        experiment_name: str,
        prediction_type: str,
        algorithm: str,
        dataset: str,
        target: str,
        test_size: float = 0.25,
        split_strategy: str = "shuffle",
    ) -> str:
        algorithm_handlers = {
            "regression": {
                "dummy": dummy.DummyRegressor,
                "linear_regression": linear_model.LinearRegression,
                "sgd": linear_model.SGDRegressor,
                "ridge": linear_model.Ridge,
                "ridge_cv": linear_model.RidgeCV,
                "elastic_net": linear_model.ElasticNet,
                "elastic_net_cv": linear_model.ElasticNetCV,
                "lasso": linear_model.Lasso,
                "lasso_cv": linear_model.LassoCV,
                "decision_tree": tree.DecisionTreeRegressor,
                "ada_boost": ensemble.AdaBoostRegressor,
                "bagging": ensemble.BaggingRegressor,
                "gradient_boosting": ensemble.GradientBoostingRegressor,
                "random_forest": ensemble.RandomForestRegressor,
                "knn": neighbors.KNeighborsRegressor,
                "mlp": neural_network.MLPRegressor,
                "svr": svm.SVR,
            },
            "classification": {
                "dummy": dummy.DummyClassifier,
                "logistic_regression": linear_model.LogisticRegression,
                "sgd": linear_model.SGDClassifier,
                "ridge": linear_model.RidgeClassifier,
                "ridge_cv": linear_model.RidgeClassifierCV,
                "decision_tree": tree.DecisionTreeClassifier,
                "ada_boost": ensemble.AdaBoostClassifier,
                "bagging": ensemble.BaggingClassifier,
                "gradient_boosting": ensemble.GradientBoostingClassifier,
                "random_forest": ensemble.RandomForestClassifier,
                "knn": neighbors.KNeighborsClassifier,
                "mlp": neural_network.MLPClassifier,
                "svc": svm.SVC,
            },
        }

        split_handlers = {
            "shuffle": model_selection.ShuffleSplit,
        }

        available_prediction_types = algorithm_handlers.keys()
        if prediction_type not in available_prediction_types:
            return json.dumps(
                {
                    "error": f"Unknown prediction type '{prediction_type}'. Available prediction types: {', '.join(available_prediction_types)}"
                }
            )

        prediction_type_algorithms = algorithm_handlers[prediction_type]
        AlgorithmHandler = prediction_type_algorithms.get(algorithm)
        if AlgorithmHandler is None:
            return json.dumps(
                {
                    "error": f"Unknown algorithm '{algorithm}'. Available algorithms for prediction type '{prediction_type}': {', '.join(prediction_type_algorithms.keys())}"
                }
            )

        SplitHandler = split_handlers.get(split_strategy)
        if SplitHandler is None:
            return json.dumps(
                {
                    "error": f"Unknown split strategy '{split_strategy}'. Available algorithms: {', '.join(split_handlers.keys())}"
                }
            )

        test_size_range = (0, 1)
        if test_size < test_size_range[0] or test_size > test_size_range[1]:
            return json.dumps(
                {
                    "error": f"Invalid test size '{test_size}'. Allowed test size range: ({test_size_range[0]}, {test_size_range[1]})"
                }
            )

        try:
            data = pd.read_sql(f"SELECT * FROM {dataset};", self.conn)
        except pd.errors.DatabaseError:
            return json.dumps({"error": f"Unknown dataset '{dataset}'"})

        try:
            X, y = data.drop([target], axis=1), data[target]
        except KeyError:
            return json.dumps(
                {
                    "error": f"Unknown target column '{target}'. Available columns: {', '.join(data.columns)}"
                }
            )

        experiment = self.get_or_create_experiment(experiment_name, prediction_type)
        experiment_id = experiment["id"]
        if experiment["prediction_type"] != prediction_type:
            return json.dumps(
                {
                    "error": f"Existing experiment {experiment} has a prediction type of '{prediction_type}'"
                }
            )

        with self.conn:
            run = self.conn.execute(
                """
                INSERT INTO sqml_runs (status, algorithm, dataset, target, test_size, split_strategy, experiment_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                RETURNING *
                """,
                (
                    "pending",
                    algorithm,
                    dataset,
                    target,
                    test_size,
                    split_strategy,
                    experiment_id,
                ),
            ).fetchone()
            run_id = run["id"]

        splitter = SplitHandler(n_splits=1, test_size=test_size)
        train_indices, test_indices = next(splitter.split(X, y))
        X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
        X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]

        estimator = pipeline.Pipeline(
            [
                ("scaler", preprocessing.StandardScaler()),
                ("model", AlgorithmHandler()),
            ]
        )
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        score = estimator.score(X_test, y_test)

        model_metrics = dict(score=score)
        if prediction_type == "classification":
            model_metrics["accuracy"] = metrics.accuracy_score(y_test, y_pred)
            model_metrics["f1"] = metrics.f1_score(y_test, y_pred, average="weighted")
            model_metrics["precision"] = metrics.precision_score(
                y_test, y_pred, average="weighted"
            )
            model_metrics["recall"] = metrics.recall_score(
                y_test, y_pred, average="weighted"
            )
        else:
            model_metrics["r2"] = metrics.r2_score(y_test, y_pred)
            model_metrics["mae"] = metrics.mean_absolute_error(y_test, y_pred)
            model_metrics["rmse"] = metrics.mean_squared_error(
                y_test, y_pred, squared=False
            )

        serialized_model = pickle.dumps(estimator)

        with self.conn:
            model = self.conn.execute(
                """
                INSERT INTO sqml_models(run_id, library, data)
                VALUES (?, ?, ?)
                RETURNING *;
                """,
                (run_id, "scikit-learn", serialized_model),
            ).fetchone()
            model_id = model["id"]

            for k, v in model_metrics.items():
                self.conn.execute(
                    """
                    INSERT INTO sqml_metrics(model_id, name, value)
                    VALUES (?, ?, ?);
                    """,
                    (model_id, k, v),
                )

            current_deployment = self.conn.execute(
                """
                SELECT
                    sqml_deployments.id AS deployment_id,
                    sqml_metrics.value AS score
                FROM sqml_deployments
                JOIN sqml_models ON sqml_models.id = sqml_deployments.model_id
                JOIN sqml_metrics ON sqml_metrics.model_id = sqml_models.id
                WHERE
                    sqml_deployments.experiment_id = ?
                    AND sqml_deployments.active = TRUE
                    AND sqml_metrics.name = 'score'
                """,
                (experiment_id,),
            ).fetchone()

            deployment_id = (
                current_deployment["deployment_id"] if current_deployment else None
            )
            initial_score = 0.0 if prediction_type == "classification" else -math.inf
            current_score = (
                current_deployment["score"] if current_deployment else initial_score
            )
            deployed = False

            if score > current_score:
                deployment = self.conn.execute(
                    """
                    INSERT INTO sqml_deployments(experiment_id, model_id)
                    VALUES (?, ?)
                    RETURNING *
                    """,
                    (experiment_id, model_id),
                ).fetchone()
                deployment_id = deployment["id"]

                self.conn.execute(
                    """
                    UPDATE sqml_deployments
                    SET active = (CASE WHEN id = :deployment_id THEN TRUE ELSE FALSE END)
                    WHERE experiment_id = :experiment_id
                    """,
                    dict(deployment_id=deployment_id, experiment_id=experiment_id),
                )

                deployed = True

            self.conn.execute(
                """
                UPDATE sqml_runs
                SET status = :status
                WHERE id = :id
                """,
                dict(id=run_id, status="success"),
            )

        return json.dumps(
            {
                "experiment_name": experiment_name,
                "prediction_type": prediction_type,
                "algorithm": algorithm,
                "deployed": deployed,
                "score": score,
            }
        )

    def predict(self, experiment_name: str, features: str) -> t.Union[float, str]:
        feature_array = pd.DataFrame([json.loads(features)])

        experiment = self.conn.execute(
            """
            SELECT *
            FROM sqml_experiments
            WHERE name = ?
            """,
            (experiment_name,),
        ).fetchone()

        if experiment is None:
            return json.dumps({"error": f"Unknown experiment '{experiment_name}'"})

        experiment_id = experiment["id"]

        deployment = self.conn.execute(
            """
            SELECT
                sqml_deployments.id AS id,
                sqml_models.id AS model_id,
                sqml_models.library AS library,
                sqml_models.data AS data
            FROM sqml_deployments
            JOIN sqml_models ON sqml_models.id = sqml_deployments.model_id
            WHERE experiment_id = ? AND active = TRUE;
            """,
            (experiment_id,),
        ).fetchone()

        if deployment is None:
            return json.dumps(
                {
                    "error": f"No deployment found for experiment '{experiment_name}'. Model must be trained successfully before running predictions"
                }
            )

        model = pickle.loads(deployment["data"])
        predictions = model.predict(feature_array)
        return float(predictions[0])

    def predict_batch(self, experiment_name: str, features: str) -> str:
        feature_matrix = pd.DataFrame(json.loads(features))

        experiment = self.conn.execute(
            """
            SELECT *
            FROM sqml_experiments
            WHERE name = ?
            """,
            (experiment_name,),
        ).fetchone()

        if experiment is None:
            return json.dumps({"error": f"Unknown experiment '{experiment_name}'"})

        experiment_id = experiment["id"]

        deployment = self.conn.execute(
            """
            SELECT
                sqml_deployments.id AS id,
                sqml_models.id AS model_id,
                sqml_models.library AS library,
                sqml_models.data AS data
            FROM sqml_deployments
            JOIN sqml_models ON sqml_models.id = sqml_deployments.model_id
            WHERE experiment_id = ? AND active = TRUE;
            """,
            (experiment_id,),
        ).fetchone()

        if deployment is None:
            return json.dumps(
                {
                    "error": f"No deployment found for experiment '{experiment_name}'. Model must be trained successfully before running predictions"
                }
            )

        model = pickle.loads(deployment["data"])
        predictions = model.predict(feature_matrix)

        return json.dumps(predictions.tolist())

    def get_or_create_experiment(
        self, name: str, prediction_type: str
    ) -> t.Mapping[str, t.Any]:
        with self.conn:
            experiment = self.conn.execute(
                """
                SELECT *
                FROM sqml_experiments
                WHERE name = ?
                """,
                (name,),
            ).fetchone()
            if experiment is None:
                experiment = self.conn.execute(
                    """
                    INSERT INTO sqml_experiments(name, prediction_type)
                    VALUES (?, ?)
                    RETURNING *
                    """,
                    (name, prediction_type),
                ).fetchone()

        return experiment
