## pyDKE 2.0.0

### Highlights

- Migrated packaging from Poetry to uv / PEP 621.
- Minimum Python version is now 3.14.
- Fixed compatibility with recent NumPy, SciPy, and CVXPY.
- Fixed SciPy spherical harmonic API migration.
- Improved IRLLS numerical stability.
- Added universal pure-Python wheel: `py3-none-any`.

### Installation

To install uv with bash terminal, create virtual environment and install pyDKE v2.0.0:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .venv --python 3.14
source .venv/bin/activate
uv pip install pydke-2.0.0-py3-none-any.whl
pydesigner --help
