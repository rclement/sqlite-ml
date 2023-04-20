import json
import sqlite3
import sys
import pytest
import sklearn
import sqlite_utils

from faker import Faker

from sqlite_ml import sqml


@pytest.fixture(scope="function")
def sqml_instance(db_conn: sqlite3.Connection) -> sqml.SQML:
    instance = sqml.SQML()
    instance.setup_schema(db_conn)
    instance.register_functions(db_conn)
    return instance


# ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql_table",
    [
        "sqml_experiments",
        "sqml_runs",
        "sqml_models",
        "sqml_metrics",
        "sqml_deployments",
    ],
)
def test_plugin_created_sql_table(
    sqml_db: sqlite_utils.Database, sqml_instance: sqml.SQML, sql_table: str
) -> None:
    assert sql_table in sqml_db.table_names()


@pytest.mark.parametrize(
    "sql_view",
    [
        "sqml_runs_overview",
    ],
)
def test_plugin_created_sql_view(
    sqml_db: sqlite_utils.Database, sqml_instance: sqml.SQML, sql_view: str
) -> None:
    assert sql_view in sqml_db.view_names()


@pytest.mark.parametrize(
    "sql_function",
    [
        "sqml_python_version",
        "sqml_sklearn_version",
        "sqml_load_dataset",
        "sqml_train",
        "sqml_predict",
    ],
)
def test_plugin_registered_sql_function(
    db_conn: sqlite3.Connection, sqml_instance: sqml.SQML, sql_function: str
) -> None:
    query = """
        SELECT * FROM pragma_function_list;
        """
    rows = db_conn.execute(query).fetchall()
    available_sql_functions = [f["name"] for f in rows]
    assert sql_function in available_sql_functions


# ------------------------------------------------------------------------------


def test_sqml_python_version(
    db_conn: sqlite3.Connection, sqml_instance: sqml.SQML
) -> None:
    query = """
        SELECT sqml_python_version() AS version;
        """
    rows = db_conn.execute(query).fetchall()

    assert len(rows) == 1
    assert rows[0]["version"] == sys.version


def test_sqml_sklearn_version(
    db_conn: sqlite3.Connection, sqml_instance: sqml.SQML
) -> None:
    query = """
        SELECT sqml_sklearn_version() AS version;
        """
    rows = db_conn.execute(query).fetchall()

    assert len(rows) == 1
    assert rows[0]["version"] == sklearn.__version__


# ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dataset",
    ["iris", "digits", "wine", "breast_cancer", "diabetes"],
)
def test_sqml_load_dataset(
    sqml_db: sqlite_utils.Database,
    db_conn: sqlite3.Connection,
    sqml_instance: sqml.SQML,
    dataset: str,
) -> None:
    query = f"""
        SELECT sqml_load_dataset('{dataset}') AS info;
        """
    rows = db_conn.execute(query).fetchall()
    assert len(rows) == 1

    info = json.loads(rows[0]["info"])
    columns = [c.name for c in sqml_db.table(info["table"]).columns]
    count = db_conn.execute(
        f'SELECT count(*) AS count FROM [{info["table"]}]',
    ).fetchone()["count"]

    assert info["table"] == f"dataset_{dataset}"
    assert columns == info["feature_names"] + ["target"]
    assert count > 0 and count == info["size"]


def test_sqml_load_dataset_unknown(
    db_conn: sqlite3.Connection,
    sqml_instance: sqml.SQML,
) -> None:
    query = """
        SELECT sqml_load_dataset('unknown') AS info;
        """
    rows = db_conn.execute(query).fetchall()
    assert len(rows) == 1

    info = json.loads(rows[0]["info"])
    assert "error" in info.keys()


# ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("prediction_type", "algorithm"),
    [
        ("regression", "dummy"),
        ("regression", "linear_regression"),
        ("regression", "sgd"),
        ("regression", "ridge"),
        ("regression", "ridge_cv"),
        ("regression", "elastic_net"),
        ("regression", "elastic_net_cv"),
        ("regression", "lasso"),
        ("regression", "lasso_cv"),
        ("regression", "decision_tree"),
        ("regression", "ada_boost"),
        ("regression", "bagging"),
        ("regression", "gradient_boosting"),
        ("regression", "random_forest"),
        ("regression", "knn"),
        ("regression", "mlp"),
        ("regression", "svr"),
        ("classification", "dummy"),
        ("classification", "logistic_regression"),
        ("classification", "sgd"),
        ("classification", "ridge"),
        ("classification", "ridge_cv"),
        ("classification", "decision_tree"),
        ("classification", "ada_boost"),
        ("classification", "bagging"),
        ("classification", "gradient_boosting"),
        ("classification", "random_forest"),
        ("classification", "knn"),
        ("classification", "mlp"),
        ("classification", "svc"),
    ],
)
def test_sqml_train(
    db_conn: sqlite3.Connection,
    sqml_instance: sqml.SQML,
    faker: Faker,
    prediction_type: str,
    algorithm: str,
) -> None:
    experiment_name = faker.bs()
    dataset = f"data_{prediction_type}"
    target = "target"
    query = f"""
        SELECT sqml_train(
            '{experiment_name}',
            '{prediction_type}',
            '{algorithm}',
            '{dataset}',
            '{target}'
        ) AS training;
        """

    rows = db_conn.execute(query).fetchall()
    assert len(rows) == 1

    training = json.loads(rows[0]["training"])

    assert training["experiment_name"] == experiment_name
    assert training["prediction_type"] == prediction_type
    assert training["algorithm"] == algorithm
    assert training["deployed"]
    assert isinstance(training["score"], float)

    experiment = db_conn.execute(
        """
        SELECT *
        FROM sqml_experiments
        WHERE id = 1
        """
    ).fetchone()
    assert experiment["name"] == experiment_name
    assert experiment["prediction_type"] == prediction_type

    run = db_conn.execute(
        """
        SELECT *
        FROM sqml_runs
        WHERE id = 1
        """
    ).fetchone()
    assert run["status"] == "success"
    assert run["algorithm"] == algorithm
    assert run["dataset"] == dataset
    assert run["target"] == target
    assert run["test_size"] == 0.25
    assert run["split_strategy"] == "shuffle"
    assert run["experiment_id"] == 1

    model = db_conn.execute(
        """
        SELECT *
        FROM sqml_models
        WHERE id = 1
        """
    ).fetchone()
    assert model["run_id"] == 1
    assert model["library"] == "scikit-learn"
    assert isinstance(model["data"], bytes) and len(model["data"]) > 0

    metrics = {
        m["name"]: m["value"]
        for m in db_conn.execute(
            """
            SELECT *
            FROM sqml_metrics
            WHERE model_id = 1
            """
        ).fetchall()
    }

    assert isinstance(metrics["score"], float)
    if prediction_type == "regression":
        assert len(metrics.keys()) == 4
        assert isinstance(metrics["r2"], float)
        assert isinstance(metrics["mae"], float)
        assert isinstance(metrics["rmse"], float)
        assert metrics["score"] == metrics["r2"]
    else:
        assert len(metrics.keys()) == 5
        assert isinstance(metrics["accuracy"], float)
        assert isinstance(metrics["f1"], float)
        assert isinstance(metrics["precision"], float)
        assert isinstance(metrics["recall"], float)
        assert metrics["score"] == metrics["accuracy"]

    deployment = db_conn.execute(
        """
        SELECT *
        FROM sqml_deployments
        WHERE id = 1
        """
    ).fetchone()
    assert deployment["experiment_id"] == 1
    assert deployment["model_id"] == 1
    assert deployment["active"]


def test_sqml_train_better_model(
    db_conn: sqlite3.Connection, sqml_instance: sqml.SQML, faker: Faker
) -> None:
    experiment_name = faker.bs()
    prediction_type = "regression"
    algorithm = "linear_regression"
    dataset = f"data_{prediction_type}"
    target = "target"
    query = f"""
        SELECT sqml_train(
            '{experiment_name}',
            '{prediction_type}',
            '{algorithm}',
            '{dataset}',
            '{target}'
        ) AS training;
        """
    db_conn.execute(query)
    db_conn.execute(
        """
        UPDATE sqml_metrics
        SET value = 0.5
        WHERE id = 1 AND name = 'score'
        """
    )

    rows = db_conn.execute(query).fetchall()
    assert len(rows) == 1

    training = json.loads(rows[0]["training"])
    assert training["deployed"]

    runs = db_conn.execute("SELECT * FROM sqml_runs ORDER BY id").fetchall()
    assert len(runs) == 2

    deployments = db_conn.execute(
        "SELECT * FROM sqml_deployments ORDER BY id"
    ).fetchall()
    assert len(deployments) == 2
    assert not deployments[0]["active"]
    assert deployments[0]["model_id"] == 1
    assert deployments[1]["active"]
    assert deployments[1]["model_id"] == 2


def test_sqml_train_worse_model(
    db_conn: sqlite3.Connection, sqml_instance: sqml.SQML, faker: Faker
) -> None:
    experiment_name = faker.bs()
    prediction_type = "regression"
    algorithm = "linear_regression"
    dataset = f"data_{prediction_type}"
    target = "target"
    query = f"""
        SELECT sqml_train(
            '{experiment_name}',
            '{prediction_type}',
            '{algorithm}',
            '{dataset}',
            '{target}'
        ) AS training;
        """
    db_conn.execute(query)
    db_conn.execute(
        """
        UPDATE sqml_metrics
        SET value = 1.0
        WHERE id = 1 AND name = 'score'
        """
    )

    rows = db_conn.execute(query).fetchall()
    assert len(rows) == 1

    training = json.loads(rows[0]["training"])
    assert not training["deployed"]

    runs = db_conn.execute("SELECT * FROM sqml_runs ORDER BY id").fetchall()
    assert len(runs) == 2

    deployments = db_conn.execute(
        "SELECT * FROM sqml_deployments ORDER BY id"
    ).fetchall()
    assert len(deployments) == 1
    assert deployments[0]["active"]
    assert deployments[0]["model_id"] == 1


def test_sqml_train_existing_experiment(
    db_conn: sqlite3.Connection, sqml_instance: sqml.SQML, faker: Faker
) -> None:
    experiment_name = faker.bs()
    prediction_type = "regression"
    algorithm = "linear_regression"
    dataset = f"data_{prediction_type}"
    target = "target"
    query = f"""
        SELECT sqml_train(
            '{experiment_name}',
            '{prediction_type}',
            '{algorithm}',
            '{dataset}',
            '{target}'
        ) AS training;
        """
    db_conn.execute(query)
    db_conn.execute(query)

    experiment = db_conn.execute(
        """
        SELECT count(*) AS count
        FROM sqml_experiments
        """
    ).fetchone()
    assert experiment["count"] == 1

    run = db_conn.execute(
        """
        SELECT count(*) AS count
        FROM sqml_runs
        """
    ).fetchone()
    assert run["count"] == 2


def test_sqml_train_existing_experiment_wrong_prediction_type(
    db_conn: sqlite3.Connection, sqml_instance: sqml.SQML, faker: Faker
) -> None:
    experiment_name = faker.bs()
    prediction_type = "regression"
    algorithm = "linear_regression"
    dataset = f"data_{prediction_type}"
    target = "target"
    query = f"""
        SELECT sqml_train(
            '{experiment_name}',
            '{prediction_type}',
            '{algorithm}',
            '{dataset}',
            '{target}'
        ) AS training;
        """
    db_conn.execute(query)

    prediction_type = "classification"
    algorithm = "logistic_regression"
    query = f"""
        SELECT sqml_train(
            '{experiment_name}',
            '{prediction_type}',
            '{algorithm}',
            '{dataset}',
            '{target}'
        ) AS training;
        """

    rows = db_conn.execute(query).fetchall()
    assert len(rows) == 1

    training = json.loads(rows[0]["training"])
    assert "error" in training.keys()


def test_sqml_train_wrong_prediction_type_algorithm(
    db_conn: sqlite3.Connection, sqml_instance: sqml.SQML, faker: Faker
) -> None:
    experiment_name = faker.bs()
    prediction_type = "regression"
    algorithm = "logistic_regression"
    dataset = f"data_{prediction_type}"
    target = "target"
    query = f"""
        SELECT sqml_train(
            '{experiment_name}',
            '{prediction_type}',
            '{algorithm}',
            '{dataset}',
            '{target}'
        ) AS training;
        """

    rows = db_conn.execute(query).fetchall()
    assert len(rows) == 1

    training = json.loads(rows[0]["training"])
    assert "error" in training.keys()


def test_sqml_train_unknown_prediction_type(
    db_conn: sqlite3.Connection, sqml_instance: sqml.SQML, faker: Faker
) -> None:
    experiment_name = faker.bs()
    prediction_type = "unknown"
    algorithm = "linear_regression"
    dataset = f"data_{prediction_type}"
    target = "target"
    query = f"""
        SELECT sqml_train(
            '{experiment_name}',
            '{prediction_type}',
            '{algorithm}',
            '{dataset}',
            '{target}'
        ) AS training;
        """

    rows = db_conn.execute(query).fetchall()
    assert len(rows) == 1

    training = json.loads(rows[0]["training"])
    assert "error" in training.keys()


def test_sqml_train_unknown_algorithm(
    db_conn: sqlite3.Connection, sqml_instance: sqml.SQML, faker: Faker
) -> None:
    experiment_name = faker.bs()
    prediction_type = "regression"
    algorithm = "unknown"
    dataset = f"data_{prediction_type}"
    target = "target"
    query = f"""
        SELECT sqml_train(
            '{experiment_name}',
            '{prediction_type}',
            '{algorithm}',
            '{dataset}',
            '{target}'
        ) AS training;
        """

    rows = db_conn.execute(query).fetchall()
    assert len(rows) == 1

    training = json.loads(rows[0]["training"])
    assert "error" in training.keys()


def test_sqml_train_unknown_dataset(
    db_conn: sqlite3.Connection, sqml_instance: sqml.SQML, faker: Faker
) -> None:
    experiment_name = faker.bs()
    prediction_type = "regression"
    algorithm = "linear_regression"
    dataset = "unknown"
    target = "target"
    query = f"""
        SELECT sqml_train(
            '{experiment_name}',
            '{prediction_type}',
            '{algorithm}',
            '{dataset}',
            '{target}'
        ) AS training;
        """

    rows = db_conn.execute(query).fetchall()
    assert len(rows) == 1

    training = json.loads(rows[0]["training"])
    assert "error" in training.keys()


def test_sqml_train_unknown_target(
    db_conn: sqlite3.Connection, sqml_instance: sqml.SQML, faker: Faker
) -> None:
    experiment_name = faker.bs()
    prediction_type = "regression"
    algorithm = "linear_regression"
    dataset = f"data_{prediction_type}"
    target = "unknown"
    query = f"""
        SELECT sqml_train(
            '{experiment_name}',
            '{prediction_type}',
            '{algorithm}',
            '{dataset}',
            '{target}'
        ) AS training;
        """

    rows = db_conn.execute(query).fetchall()
    assert len(rows) == 1

    training = json.loads(rows[0]["training"])
    assert "error" in training.keys()


def test_sqml_train_unknown_split_strategy(
    db_conn: sqlite3.Connection, sqml_instance: sqml.SQML, faker: Faker
) -> None:
    experiment_name = faker.bs()
    prediction_type = "regression"
    algorithm = "linear_regression"
    dataset = f"data_{prediction_type}"
    target = "target"
    test_size = 0.25
    split_strategy = "unknown"
    query = f"""
        SELECT sqml_train(
            '{experiment_name}',
            '{prediction_type}',
            '{algorithm}',
            '{dataset}',
            '{target}',
            {test_size},
            '{split_strategy}'
        ) AS training;
        """

    rows = db_conn.execute(query).fetchall()
    assert len(rows) == 1

    training = json.loads(rows[0]["training"])
    assert "error" in training.keys()


@pytest.mark.parametrize(
    "test_size",
    [-0.25, 1.1],
)
def test_sqml_train_out_of_range_test_size(
    db_conn: sqlite3.Connection,
    sqml_instance: sqml.SQML,
    faker: Faker,
    test_size: float,
) -> None:
    experiment_name = faker.bs()
    prediction_type = "regression"
    algorithm = "linear_regression"
    dataset = f"data_{prediction_type}"
    target = "target"
    split_strategy = "shuffle"
    query = f"""
        SELECT sqml_train(
            '{experiment_name}',
            '{prediction_type}',
            '{algorithm}',
            '{dataset}',
            '{target}',
            {test_size},
            '{split_strategy}'
        ) AS training;
        """

    rows = db_conn.execute(query).fetchall()
    assert len(rows) == 1

    training = json.loads(rows[0]["training"])
    assert "error" in training.keys()


# ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("prediction_type", "algorithm"),
    [
        ("regression", "linear_regression"),
        ("classification", "logistic_regression"),
    ],
)
def test_sqml_predict(
    db_conn: sqlite3.Connection,
    sqml_instance: sqml.SQML,
    faker: Faker,
    prediction_type: str,
    algorithm: str,
) -> None:
    experiment_name = faker.bs()
    dataset = f"data_{prediction_type}"
    target = "target"
    query = f"""
        SELECT sqml_train(
            '{experiment_name}',
            '{prediction_type}',
            '{algorithm}',
            '{dataset}',
            '{target}'
        ) AS training;
        """
    db_conn.execute(query)

    data_row = db_conn.execute(
        f"""
        SELECT *
        FROM {dataset}
        LIMIT 1
        """,
    ).fetchone()

    features = json.dumps({k: v for k, v in dict(data_row).items() if k != target})
    query = f"""
        SELECT sqml_predict(
            '{experiment_name}',
            '{features}'
        ) AS prediction;
        """

    rows = db_conn.execute(query).fetchall()
    assert len(rows) == 1

    prediction = rows[0]["prediction"]
    assert isinstance(prediction, float)


def test_sqml_predict_unknown_experiment(
    db_conn: sqlite3.Connection, sqml_instance: sqml.SQML, faker: Faker
) -> None:
    experiment_name = faker.bs()
    query = f"""
        SELECT sqml_predict(
            '{experiment_name}',
            '{{}}'
        ) AS prediction;
        """

    rows = db_conn.execute(query).fetchall()
    assert len(rows) == 1

    prediction = json.loads(rows[0]["prediction"])
    assert "error" in prediction.keys()


def test_sqml_predict_no_deployment(
    db_conn: sqlite3.Connection, sqml_instance: sqml.SQML, faker: Faker
) -> None:
    experiment_name = faker.bs()
    db_conn.execute(
        """
        INSERT INTO sqml_experiments(name, prediction_type)
        VALUES (?, ?)
        """,
        (experiment_name, "classification"),
    )
    query = f"""
        SELECT sqml_predict(
            '{experiment_name}',
            '{{}}'
        ) AS prediction;
        """

    rows = db_conn.execute(query).fetchall()
    assert len(rows) == 1

    prediction = json.loads(rows[0]["prediction"])
    assert "error" in prediction.keys()


# ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("prediction_type", "algorithm"),
    [
        ("regression", "linear_regression"),
        ("classification", "logistic_regression"),
    ],
)
def test_sqml_predict_batch(
    db_conn: sqlite3.Connection,
    sqml_instance: sqml.SQML,
    faker: Faker,
    prediction_type: str,
    algorithm: str,
) -> None:
    experiment_name = faker.bs()
    dataset = f"data_{prediction_type}"
    target = "target"
    query = f"""
        SELECT sqml_train(
            '{experiment_name}',
            '{prediction_type}',
            '{algorithm}',
            '{dataset}',
            '{target}'
        ) AS training;
        """
    db_conn.execute(query)

    data_rows = db_conn.execute(f"SELECT * FROM {dataset}").fetchall()
    count_rows = db_conn.execute(f"SELECT count(*) AS count FROM {dataset}").fetchone()[
        "count"
    ]

    features = ", ".join(
        f"'{k}', [{k}]" for k in dict(data_rows[0]).keys() if k != target
    )
    query = f"""
        SELECT sqml_predict_batch(
            '{experiment_name}',
            json_group_array(
                json_object({features})
            )
        ) AS predictions
        FROM {dataset};
        """

    rows = db_conn.execute(query).fetchall()
    assert len(rows) == 1

    predictions = json.loads(rows[0]["predictions"])
    assert len(predictions) == count_rows
    for pred in predictions:
        assert isinstance(pred, float)


def test_sqml_predict_batch_unknown_experiment(
    db_conn: sqlite3.Connection, sqml_instance: sqml.SQML, faker: Faker
) -> None:
    experiment_name = faker.bs()
    query = f"""
        SELECT sqml_predict_batch(
            '{experiment_name}',
            '[]'
        ) AS prediction;
        """

    rows = db_conn.execute(query).fetchall()
    assert len(rows) == 1

    prediction = json.loads(rows[0]["prediction"])
    assert "error" in prediction.keys()


def test_sqml_predict_batch_no_deployment(
    db_conn: sqlite3.Connection, sqml_instance: sqml.SQML, faker: Faker
) -> None:
    experiment_name = faker.bs()
    db_conn.execute(
        """
        INSERT INTO sqml_experiments(name, prediction_type)
        VALUES (?, ?)
        """,
        (experiment_name, "classification"),
    )

    query = f"""
        SELECT sqml_predict_batch(
            '{experiment_name}',
            '[]'
        ) AS prediction;
        """

    rows = db_conn.execute(query).fetchall()
    assert len(rows) == 1

    prediction = json.loads(rows[0]["prediction"])
    assert "error" in prediction.keys()
