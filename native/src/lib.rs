use pyo3::prelude::*;
use serde_json::json;
use sqlite3_ext::{*, function::*, vtab::*};

#[sqlite3_ext_fn(n_args=0)]
pub fn python_version(ctx: &Context, _args: &mut [&mut ValueRef]) -> Result<()> {
    let mut version = String::new();

    Python::with_gil(|py| {
        let sys = PyModule::import(py, "sys").unwrap();
        version = sys.getattr("version").unwrap().extract().unwrap();
    });

    ctx.set_result(version)
}

#[sqlite3_ext_fn(n_args=0)]
pub fn sklearn_version(ctx: &Context, _args: &mut [&mut ValueRef]) -> Result<()> {
    let mut version = String::new();

    Python::with_gil(|py| {
        let sklearn = PyModule::import(py, "sklearn").unwrap();
        version = sklearn.getattr("__version__").unwrap().extract().unwrap();
    });

    ctx.set_result(version)
}

#[sqlite3_ext_fn(n_args=0)]
pub fn count_experiments(ctx: &Context, _args: &mut [&mut ValueRef]) -> Result<()> {
    let conn = ctx.db();
    let mut results = conn.query("SELECT count(*) FROM sqml_experiments", ())?;
    let count = if let Some(row) = results.next()? {
        row[0].get_i64()
    } else {
        -1
    };
    ctx.set_result(count)
}

fn load_iris_dataset(conn: &Connection) -> (String, i64) {
    conn.execute("DROP TABLE IF EXISTS dataset_iris", ()).unwrap();
    conn.execute(
        "CREATE TABLE dataset_iris (
            sepal_length REAL,
            sepal_width REAL,
            petal_length REAL,
            petal_width REAL,
            target INTEGER
        )",
        (),
    ).unwrap();

    Python::with_gil(|py| {
        let datasets = PyModule::import(py, "sklearn.datasets").unwrap();
        let load_iris = datasets.getattr("load_iris").unwrap();
        let data = load_iris.call0().unwrap();
        let feature_names: Vec<String> = data.getattr("feature_names").unwrap().extract().unwrap();
        println!("{:?}", feature_names);
    });

    ("dataset_iris".to_string(), 0)
}

#[sqlite3_ext_fn(n_args=1)]
pub fn load_dataset(ctx: &Context, args: &mut [&mut ValueRef]) -> Result<()> {
    let dataset_name = args.get_mut(0).expect("").get_str()?;
    let conn = ctx.db();

    let (name, rows) = match dataset_name {
        "iris" => load_iris_dataset(conn),
        _ => ("unknown dataset".to_string(), 0)
    };

    let result = json!({
        "table": name,
        "rows": rows,
    });
    ctx.set_result(result.to_string())
}

#[sqlite3_ext_vtab(EponymousModule)]
struct SqmlLoadDataset {}

impl VTab<'_> for SqmlLoadDataset {
    type Aux = ();
    type Cursor = SqmlLoadDatasetCursor;

    fn connect(db: &VTabConnection, _aux: &Self::Aux, _args: &[&str]) -> Result<(String, Self)> {
        db.set_risk_level(RiskLevel::Innocuous);
        Ok((
            "CREATE TABLE x ( table, size, dataset HIDDEN )".to_owned(),
            SqmlLoadDataset {},
        ))
    }

    fn open(&self) -> Result<Self::Cursor> {
        Ok(SqmlLoadDatasetCursor::default())
    }

    fn best_index<'vtab>(&'vtab self, index_info: &mut IndexInfo) -> Result<()> {
        index_info.set_index_num(0);
        Ok(())
    }
}

#[derive(Default, Debug)]
struct SqmlLoadDatasetCursor {
    rowid: i64,
    table: String,
    size: i64,
}

impl VTabCursor<'_> for SqmlLoadDatasetCursor {
    fn filter(
        &mut self,
        index_num: i32,
        index_str: Option<&str>,
        args: &mut [&mut ValueRef],
    ) -> Result<()> {
        self.rowid = 1;
        self.table = "dataset_".to_owned();
        self.size = 123;
        Ok(())
    }

    fn next(&mut self) -> Result<()> {
        Ok(())
    }

    fn eof(&self) -> bool {
        self.rowid > 1
    }

    fn column(&self, idx: usize, ctx: &ColumnContext) -> Result<()> {
        Ok(())
    }

    fn rowid(&self) -> Result<i64> {
        Ok(self.rowid)
    }
}

#[sqlite3_ext_main]
fn init(db: &Connection) -> Result<()> {
    db.create_scalar_function("python_version", &PYTHON_VERSION_OPTS, python_version)?;
    db.create_scalar_function("sklearn_version", &SKLEARN_VERSION_OPTS, sklearn_version)?;
    db.create_scalar_function("count_experiments", &COUNT_EXPERIMENTS_OPTS, count_experiments)?;
    db.create_scalar_function("sqml_load_dataset", &LOAD_DATASET_OPTS, load_dataset)?;
    // db.create_module("sqml_load_dataset", SqmlLoadDataset::module(), ())?;

    let schema = include_str!("schema.sql");
    db.execute(&schema, ())?;

    Ok(())
}
