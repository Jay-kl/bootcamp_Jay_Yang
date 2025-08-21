This project demonstrates how to:
- Create a sample stock price DataFrame with pandas
- Save it in CSV and Parquet formats
- Reload both files
- Validate that they match in shape and data types


Feature
1. Automatic directory creation when saving files
2. Automatic format detection based on file suffix
3. Timestamped filenames to avoid overwrite

Folder Structured
1. Code in base folder (`homework_stage05.ipynb`)
2. `data/raw/` to save CSV (e.g., `stock_data_YYYYMMDD_HHMMSS.csv`, `util_*.csv`)
3. `data/processed/` to save Parquet (e.g., `stock_data_YYYYMMDD_HHMMSS.parquet`, `util_*.parquet`)

Utilities
- `detect_format(path)`: detect format by suffix (`.csv`, `.parquet`)
- `write_df(df, path)`: create parent dirs; write CSV or Parquet (friendly error if parquet engine missing)
- `read_df(path)`: read CSV (auto `parse_dates=['date']` when `date` exists) or Parquet

Validation
- `shape_equal`: original vs reloaded have the same shape
- `date_is_datetime`: `date` column is datetime dtype
- `price_is_numeric`: `price` column is numeric dtype
Example output:
- `{'shape_equal': True, 'date_is_datetime': True, 'price_is_numeric': True}`

Assumptions / Notes
- Filenames use timestamp to avoid overwrite; change to fixed names if you need idempotent outputs
- CSV has no native types; we parse `date` if present to ensure datetime
- Parquet engine default is `pyarrow`
- Example data is small and synthetic; adapt schema checks for real projects

How to run
1. Create `.env` with `DATA_DIR_RAW` and `DATA_DIR_PROCESSED`
2. `pip install -U pandas python-dotenv pyarrow fastparquet`
3. Run `homework_stage05.ipynb` to save, reload, and validate
4. Check generated files under `data/raw/` and `data/processed/`