.load target/debug/libsqliteml
.tables

SELECT 'Python version: ' || python_version();
SELECT 'Scikit-Learn version: ' || sklearn_version();
SELECT 'Count experiments: ' || count_experiments();

SELECT sqml_load_dataset('iris');
-- SELECT * FROM sqml_load_dataset('iris');

.schema dataset_iris
SELECT count(*) FROM dataset_iris;
