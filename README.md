# dpitg
Source code and notebooks for the paper *"Precision and Decisiveness as Goals: Reliable Sequential Testing with a Dual Stopping Criterion"* by Eyal A. Kazin.


## Setup

### 1. Create a virtual environment

```bash
python3.10 -m venv .venv
```

### 2. Activate the virtual environment

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter

```bash
jupyter notebook
```


## Cache

Long runs compute and store HDI values in a `BinaryAccounting` cache to avoid redundant work.
Cache files are saved as `.pkl` files under the `cache/` directory (auto-created on first use,
git-ignored). Pass a filename when initialising:

```python
from utils_experiments_binomial import BinaryAccounting
ba = BinaryAccounting.load_or_create("cache/my_run")  # loads if exists, creates fresh otherwise
# ... run experiments ...
ba.save()
```

Cache files can reach tens or hundreds of MB — do not commit them to the repository.

## Testing

Tests live in the `tests/` directory and use [pytest](https://docs.pytest.org/).

### Install pytest (one-time)

```bash
pip install pytest
```

### Run all tests

```bash
python -m pytest tests/ -v
```
