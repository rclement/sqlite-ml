CREATE TABLE IF NOT EXISTS sqml_experiments (
    id INTEGER PRIMARY KEY,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    name TEXT NOT NULL,
    prediction_type  TEXT NOT NULL,
    UNIQUE(name)
);

CREATE TABLE IF NOT EXISTS sqml_runs (
    id INTEGER PRIMARY KEY,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    status TEXT NOT NULL,
    algorithm TEXT NOT NULL,
    dataset TEXT NOT NULL,
    target TEXT NOT NULL,
    test_size REAL NOT NULL,
    split_strategy TEXT NOT NULL,
    experiment_id INTEGER NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES sqml_experiments(id)
);

CREATE TABLE IF NOT EXISTS sqml_models (
    id INTEGER PRIMARY KEY,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    run_id INTEGER NOT NULL,
    library TEXT NOT NULL,
    data BLOB NOT NULL,
    FOREIGN KEY (run_id) REFERENCES sqml_runs(id)
);

CREATE TABLE IF NOT EXISTS sqml_metrics (
    id INTEGER PRIMARY KEY,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    model_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    value REAL NOT NULL,
    FOREIGN KEY (model_id) REFERENCES sqml_models(id)
);

CREATE TABLE IF NOT EXISTS sqml_deployments (
    id INTEGER PRIMARY KEY,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    experiment_id INTEGER,
    model_id INTEGER,
    active BOOLEAN NOT NULL DEFAULT FALSE,
    FOREIGN KEY (experiment_id) REFERENCES sqml_experiments(id),
    FOREIGN KEY (model_id) REFERENCES sqml_models(id)
);

CREATE VIEW IF NOT EXISTS sqml_runs_overview AS
SELECT
    sqml_runs.id AS run_id,
    sqml_experiments.name AS experiment,
    sqml_runs.created_at AS start_time,
    sqml_runs.status AS status,
    sqml_runs.algorithm AS algorithm,
    json_extract(
        json_group_object(
            sqml_metrics.name, sqml_metrics.value
        ) FILTER (
            WHERE sqml_metrics.name = 'score'
        ),
        '$.score'
    ) AS score
FROM
  sqml_runs
  JOIN sqml_experiments ON sqml_experiments.id = sqml_runs.experiment_id
  LEFT JOIN sqml_models ON sqml_models.run_id = sqml_runs.id
  LEFT JOIN sqml_metrics ON sqml_metrics.model_id = sqml_models.id
GROUP BY
  sqml_runs.id;