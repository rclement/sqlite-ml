# sqlite-ml

> An SQLite extension for machine learning

Train machine learning models and run predictions directly from your SQLite database.
Inspired by [PostgresML](https://postgresml.org).

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
