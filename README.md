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
