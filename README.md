# sqlite-ml

> An SQLite extension for machine learning

Train machine learning models and run predictions directly from your SQLite database.
Inspired by [PostgresML](https://postgresml.org).

[![PyPI](https://img.shields.io/pypi/v/sqlite-ml.svg)](https://pypi.org/project/sqlite-ml/)
[![CI/CD](https://github.com/rclement/sqlite-ml/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/rclement/sqlite-ml/actions/workflows/ci-cd.yml)
[![Coverage Status](https://img.shields.io/codecov/c/github/rclement/sqlite-ml)](https://codecov.io/gh/rclement/sqlite-ml)
[![License](https://img.shields.io/github/license/rclement/sqlite-ml)](https://github.com/rclement/sqlite-ml/blob/master/LICENSE)

## Why?

Why bother running Machine Learning workloads in SQLite? Good question!
Here are some answers:

- Machine Learning number one problem is data
- Instead of trying to bring the data to the model, why not bring a model along the data?
- Instead of exporting the data some place else, training a model, performing inference and bringing back the prediction into the database, why not do that directly alonside the data?
- Lots of ETL/ELT workloads are converging to pure SQL processing, why not do that also with predictions?
- The field of MLOps tries to unify the ML lifecycle but this is hard when working on multiple environments
- SQLite is fast
- The SQLite ecosystem is quite good (take a look at all the wonderful things around Datasette)
- SQLite is often used for ad-hoc data analysis, why not give the opportunity to also make predictions at the same time?
- Easy to integrate predictions for existings applications, use a simple SQL query!

## Install

`pip install sqlite-ml`

## Warning

Currently, the only way to load this extension is through **Python SQLite3**:

```python
import sqlite3

from sqlite_ml.sqml import SQML

# get a `sqlite3.Connection` object with read-write permissions
conn = sqlite3.connect(":memory:")

# setup sqlite-ml extension
sqml = SQML()
sqml.setup_schema(conn)
sqml.register_functions(conn)

# execute sqlite-ml functions
conn.execute("SELECT sqml_python_version();").fetchone()[0]
```

We are working on making this extension a native SQLite extension,
usable within any SQLite context, stay tuned!

## Tutorial

Using `sqlite-ml` you can start training Machine Learning models directly
along your data, simply by using custom SQL functions! Let's get started by
training a classifier against the famous "Iris Dataset" to predict flower types.

### Loading the dataset

First let's load our data. For a real world project, your data may live with its
own table or being accessed through an SQL view. For the purpose of this tutorial,
we can use the `sqml_load_dataset` function to load
[standard Scikit-Learn datasets](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets):

```sql
SELECT sqml_load_dataset('iris') AS dataset;
```

It will return the following data:

| dataset |
| --- |
| {"table": "dataset_iris", "feature_names": ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"], "target_names": ["setosa", "versicolor", "virginica"], "size": 150} |

The Iris dataset is loaded into a table nammed `dataset_iris`,
containing 150 examples, 4 features and 3 classes to be predicted.

### Training a classifier

Now that our dataset is ready, let's train a first machine learning model to
perform a classification task using the `sqml_train` function:

```sql
SELECT sqml_train(
  'Iris prediction',
  'classification',
  'logistic_regression',
  'dataset_iris',
  'target'
) AS training;
```

It will return the following data:

| training |
| --- |
| {"experiment_name": "Iris prediction", "prediction_type": "classification", "algorithm": "logistic_regression", "deployed": true, "score": 0.9473684210526315} |

We have just trained our first machine learning model! The output data informs us
that our model has been trained, yields a score of 0.94 and has been deployed.

### Performing predictions

Now that we have trained our classifier, let's use it to make predictions!

Predict the target label for the first row of `dataset_iris` using the
`sqml_predict` function:

```sql
SELECT
  dataset_iris.*,
  sqml_predict(
    'Iris prediction',
    json_object(
      'sepal length (cm)', [sepal length (cm)],
      'sepal width (cm)', [sepal width (cm)],
      'petal length (cm)', [petal length (cm)],
      'petal width (cm)', [petal width (cm)]
    )
  ) AS prediction
FROM dataset_iris
LIMIT 1;
```

This will output the following data:

| sepal length (cm) | sepal width (cm) | petal length (cm) | petal width (cm) | target | prediction |
| --- | --- | --- | --- | --- | --- |
| 5.1 | 3.5 | 1.4 | 0.2 | 0.0 | 0.0 |

Yay! Our prediction is matching the target label!

Let's see if we can find some predictions not matching the target label.
To perform lots of predictions, we will use `sqml_predict_batch` which is more
efficient than `sqml_predict`:

```sql
SELECT
  dataset_iris.*,
  batch.value AS prediction,
  dataset_iris.target = batch.value AS match
FROM
  dataset_iris
  JOIN json_each (
    (
      SELECT
        sqml_predict_batch(
          'Iris prediction',
          json_group_array(
            json_object(
              'sepal length (cm)', [sepal length (cm)],
              'sepal width (cm)', [sepal width (cm)],
              'petal length (cm)', [petal length (cm)],
              'petal width (cm)', [petal width (cm)]
            )
          )
        )
      FROM
        dataset_iris
    )
  ) batch ON (batch.rowid + 1) = dataset_iris.rowid
WHERE match = FALSE;
```

This will yield the following output data:

| sepal length (cm) | sepal width (cm) | petal length (cm) | petal width (cm) | target | prediction | match |
| --- | --- | --- | --- | --- | --- | --- |
| 5.9 | 3.2 | 4.8 | 1.8 | 1.0 | 2.0 | 0 |
| 6.7 | 3.0 | 5.0 | 1.7 | 1.0 | 2.0 | 0 |
| 6.0 | 2.7 | 5.1 | 1.6 | 1.0 | 2.0 | 0 |
| 4.9 | 2.5 | 4.5 | 1.7 | 2.0 | 1.0 | 0 |

Oh no! 4 predictions have not predicted the correct target label!

Let's see if we can train a better algorithm to enhance the prediction quality.

### Training a new model

Let's use a Support Vector Machine algorithm, usually yielding better results
compared to the more simplistic Logistic Regression:

```sql
SELECT sqml_train(
  'Iris prediction',
  'classification',
  'svc',
  'dataset_iris',
  'target'
) AS training;
```

This will yield the following data:

| training |
| --- |
| {"experiment_name": "Iris prediction", "prediction_type": "classification", "algorithm": "svc", "deployed": true, "score": 0.9736842105263158} |

We can already see that the score of this new model is higher than the previous one and it has been deployed.

Let's try our new classifier on the same dataset:

```sql
SELECT
  dataset_iris.*,
  batch.value AS prediction,
  dataset_iris.target = batch.value AS match
FROM
  dataset_iris
  JOIN json_each (
    (
      SELECT
        sqml_predict_batch(
          'Iris prediction',
          json_group_array(
            json_object(
              'sepal length (cm)', [sepal length (cm)],
              'sepal width (cm)', [sepal width (cm)],
              'petal length (cm)', [petal length (cm)],
              'petal width (cm)', [petal width (cm)]
            )
          )
        )
      FROM
        dataset_iris
    )
  ) batch ON (batch.rowid + 1) = dataset_iris.rowid
WHERE match = FALSE;
```

This will lead the following results:

| sepal length (cm) | sepal width (cm) | petal length (cm) | petal width (cm) | target | prediction | match |
| --- | --- | --- | --- | --- | --- | --- |
| 5.9 | 3.2 | 4.8 | 1.8 | 1.0 | 2.0 | 0 |
| 6.7 | 3.0 | 5.0 | 1.7 | 1.0 | 2.0 | 0 |
| 6.0 | 2.7 | 5.1 | 1.6 | 1.0 | 2.0 | 0 |

Yay! We manage to predict one more target label with this new model!

Also note that we did not have to do anything to switch to the better model:
exactly the same query is used to perform the prediction without having to
specify anything about the new model! This is because new models are deployed
automatically for the current experiment only if their score outperforms the
score of the previously deployed model.

### SQL functions

This plugin registers a few SQL functions to perform machine learning model training and predictions:

`sqml_load_dataset(name, table)`
- `name: str`: name of the dataset to load
- `table: str`: (optional) custom table name destination for the dataset

`sqml_train(experiment_name, prediction_type, algorithm, dataset, target, test_size, split_strategy)`:
- `experiment_name: str`: name of the experiment to train the model within
- `prediction_type: str`: prediction task type to be performed for this experiment (`regression`, `classification`)
- `algorithm: str`: algorithm type to be trained
- `dataset: str`: name of the table or view containing the dataset
- `target: str`: name of the column to be treated as target label
- `test_size: float`: (optional) dataset test size ratio (default is `0.25`)
- `split_strategy: str`: (optional) dataset train/test split strategy (default is `shuffle`)

`sqml_predict(experiment_name, features)`
- `experiment_name: str`: name of the experiment to train the model within
- `features: json object`: JSON object containing the features

`sqml_predict_batch(experiment_name, features)`
- `experiment_name: str`: name of the experiment to train the model within
- `features: json list`: JSON list containing all feature objects

## Development

To set up this plugin locally, first checkout the code.
Then create a new virtual environment and the required dependencies:

```bash
poetry shell
poetry install
```

To run the QA suite:

```bash
black --check sqlite_ml tests
flake8 sqlite_ml tests
mypy sqlite_ml tests
pytest -v --cov=sqlite_ml --cov=tests --cov-branch --cov-report=term-missing tests
```

## Inspiration

All the things on the internet that has inspired this project:

- [PostgresML](https://postgresml.org)
- [SQLite  Run-Time Loadable Extensions](https://www.sqlite.org/loadext.html)
- [Alex Garcia's `sqlite-loadable-rs`](https://github.com/asg017/sqlite-loadable-rs)
- [Alex Garcia's SQLite extensions](https://github.com/asg017)
- [Alex Garcia, "Making SQLite extensions pip install-able"](https://observablehq.com/@asg017/making-sqlite-extensions-pip-install-able)
- [Ryan Patterson's `sqlite3_ext`](https://github.com/CGamesPlay/sqlite3_ext)
- [Max Halford, "Online gradient descent written in SQL"](https://maxhalford.github.io/blog/ogd-in-sql/)
- [Ricardo Anderegg, "Extending SQLite with Rust"](https://ricardoanderegg.com/posts/extending-sqlite-with-rust/)
- [Who needs MLflow when you have SQLite?](https://ploomber.io/blog/experiment-tracking/)
- [PostgresML is Moving to Rust for our 2.0 Release](https://postgresml.org/blog/postgresml-is-moving-to-rust-for-our-2.0-release)
- [The Virtual Table Mechanism Of SQLite](https://www.sqlite.org/vtab.html)
- [PyO3](https://pyo3.rs)
- [NoiSQL — Generating Music With SQL Queries](https://github.com/ClickHouse/NoiSQL)

## License

Licensed under Apache License, Version 2.0

Copyright (c) 2023 - present Romain Clement
