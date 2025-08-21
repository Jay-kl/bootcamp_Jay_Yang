### Stage 06 — Data Preprocessing

This homework focuses on common data cleaning and feature scaling steps, aiming to transform raw data into an analysis- and modeling-ready dataset. You can use the utility functions in `src/cleaning.py`, or follow the full workflow in `homework_stage06.ipynb`.

---

### Folder Structure
- `homework_stage06.ipynb`: Main notebook with step-by-step instructions and examples
- `src/cleaning.py`: Utility functions for data cleaning and scaling
- `data/raw_data_create.csv`: Example raw data
- `data/cleaned_data.csv`: Cleaned output (created/updated after running the notebook or code)

---

### Environment & Dependencies
- Python 3.9+
- pandas, numpy, scikit-learn

Install (recommended inside a virtual environment):
```bash
pip install -U pandas numpy scikit-learn
```

---

### Core Functions (in `src/cleaning.py`)
- `fill_missing_median(df, columns=None)`:
  - Fills numeric columns' missing values with each column's median. If `columns` is not provided, all numeric columns are used.

- `drop_missing(df, columns=None, threshold=None)`:
  - If `columns` is provided, drops rows that have NA in any of those columns.
  - Else if `threshold` is provided (0–1), keeps rows with at least `threshold * num_columns` non-NA values.
  - Else, drops rows that have any NA across all columns.

- `normalize_data(df, columns=None, method='minmax')`:
  - Scales selected (or all numeric) columns using either `minmax` (0–1) or `standard` (z-score) scaling.

- `fill_missing_general(df)`:
  - Numeric columns: fill with median
  - Object (string-like) columns: fill with `'unknown'`
  - Datetime columns: forward-fill then backward-fill

All functions return a new DataFrame and do not modify the input in place.

---

### Design Notes & Assumptions
- Median imputation is used by default for numeric columns to reduce sensitivity to outliers.
- Row dropping can be controlled by specific `columns` or a global completeness `threshold`.
- Scaling choice depends on downstream models: try `minmax` or `standard` as appropriate.
- String-like columns use a unified placeholder `'unknown'` when missing; datetime columns are filled respecting temporal order.

---
