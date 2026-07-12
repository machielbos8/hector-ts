# Example 8 — Toeplitz Factorisation: Levinson vs. Generalised Schur

A self-contained Jupyter notebook comparing the two algorithms at the heart
of Hector's speed advantage.

Every GGM covariance matrix is a symmetric positive-definite Toeplitz matrix.
Hector exploits this structure to factorise the matrix without ever forming it
explicitly.  Two algorithms are compared:

| Algorithm | Complexity | Hector class |
|-----------|-----------|-------------|
| Durbin–Levinson | O(n²) | `Levinson` |
| Generalised Schur (GSA) | O(n log² n) | `Schur` |

The notebook builds a GGM flicker-noise covariance vector of length 10 000,
runs both algorithms, verifies the result against the Gohberg–Semencul formula
(C⁻¹ reconstructed from `l1`, `l2`, δ), and produces timing curves over a
range of series lengths.  No data files are needed — everything is computed
from the GGM parameters.

## Running the notebook

```bash
pip install jupyter   # if not already installed
jupyter notebook
```

Then open `toeplitz_factorisation.ipynb`.
