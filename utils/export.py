"""
EduGuard AI — Export Module
Converts dataframes to downloadable CSV/Excel formats with formatting options.
"""
import io
import pandas as pd

def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Convert pandas DataFrame to UTF-8 encoded CSV bytes for Streamlit download."""
    return df.to_csv(index=False).encode("utf-8")

def dataframe_to_excel_bytes(df: pd.DataFrame) -> bytes:
    """Convert pandas DataFrame to Excel bytes for download."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="EduGuard_Export")
    return output.getvalue()
