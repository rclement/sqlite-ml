-- load sample datasets
SELECT sqml_load_dataset('iris');
SELECT sqml_load_dataset('digits');
SELECT sqml_load_dataset('wine');
SELECT sqml_load_dataset('breast_cancer');
SELECT sqml_load_dataset('diabetes');

-- train some models
SELECT sqml_train('Iris prediction', 'classification', 'logistic_regression', 'dataset_iris', 'target');
SELECT sqml_train('Iris prediction', 'classification', 'svc', 'dataset_iris', 'target');
SELECT sqml_train('Digits prediction', 'classification', 'logistic_regression', 'dataset_digits', 'target');
SELECT sqml_train('Digits prediction', 'classification', 'svc', 'dataset_digits', 'target');
SELECT sqml_train('Diabetes prediction', 'regression', 'linear_regression', 'dataset_diabetes', 'target');
SELECT sqml_train('Diabetes prediction', 'regression', 'svr', 'dataset_diabetes', 'target');

-- predict with a single row
SELECT sqml_predict(
    'Iris prediction',
    (
        SELECT json_object(
            'sepal length (cm)', [sepal length (cm)],
            'sepal width (cm)', [sepal width (cm)],
            'petal length (cm)', [petal length (cm)],
            'petal width (cm)', [petal width (cm)]
        )
        FROM dataset_iris
        LIMIT 1
    )
) AS prediction;

SELECT sqml_predict(
    'Diabetes prediction',
    (
        SELECT json_array([age], [sex], [bmi], [bp], [s1], [s2], [s3], [s4], [s5], [s6])
        FROM dataset_diabetes
        LIMIT 1
    )
) AS prediction;

-- predict for a complete dataset and compare results
SELECT
    *,
    target = prediction AS match
FROM (
    SELECT
        *,
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
)
ORDER BY match;

SELECT
    *,
    abs(target - prediction) AS abs_error
FROM (
    SELECT
        *,
        sqml_predict(
            'Diabetes prediction',
            json_array([age], [sex], [bmi], [bp], [s1], [s2], [s3], [s4], [s5], [s6])
        ) AS prediction
    FROM dataset_diabetes
)
ORDER BY abs_error DESC;

-- predict in batch
SELECT
    *,
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
    ) AS prediction
FROM dataset_iris;

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
ORDER BY match;
