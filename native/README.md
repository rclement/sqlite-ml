# sqlite-ml (native)

## Usage

```
env PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install 3.11.3
pyenv local 3.11.3
poetry install
poetry shell

cargo build

export PYTHONPATH=$(python -c 'import site; print(site.getsitepackages()[0])')
export PATH="/usr/local/opt/sqlite/bin:$PATH"
sqlite3 test.db < test.sql
```

```
CREATE VIRTUAL TABLE linreg
USING ml_model(
    dataset=mytable,
    targets=['label'],
    algorithm='sklearn_linear_regression',
    train_ratio=0.7,
    split_method='shuffle'
);
```

## Caveats

### Python module not found

When importing Python modules from Rust within a virtual environment on MacOS,
modules are currently not found by the runtime embedded by PyO3.

E.g. `ModuleNotFoundError("No module named 'sklearn'")`

The issue seem to lie with how PyO3 initialize the Python interpreter.

Issue on GitHub: https://github.com/PyO3/pyo3/issues/1741

Workaround: set `PYTHONPATH` with virtualenv site-package folder using
`python -c 'import site; print(site.getsitepackages()[0])'`.
