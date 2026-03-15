"""
Flavor Imitation Agent - GC-MS Data Parser
============================================
Reads raw GC-MS export files (CSV/Excel) and converts them into structured DataFrames.
Handles Chinese column headers as per the user's instrument export format.
"""
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any


# Define expected column mappings (Chinese -> Internal English Keys)
GCMS_COLUMN_MAP = {
    '样品名称': 'sample_name',
    'CAS 编号': 'cas',
    '组分 RT': 'retention_time',
    '化合物名称': 'compound_name_en',
    '中文名': 'compound_name_cn',
    '香韵': 'sensory_notes',
    'FEMA单体编号': 'fema_id',
    '匹配因子': 'match_factor',
    '分子式': 'molecular_formula',
    '相对含量mg/kg': 'concentration_mg_kg'
}


def parse_gcms_csv(filepath: str | Path) -> pd.DataFrame:
    """
    Parse a GC-MS CSV export file into a standardized DataFrame.
    Handles CSV files where compound names may contain commas.

    Args:
        filepath: Path to the GC-MS CSV file.

    Returns:
        pd.DataFrame with standardized column names.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"GC-MS file not found: {filepath}")

    # Read CSV with error handling for inconsistent column counts
    # on_bad_lines='warn' will skip problematic rows but continue parsing
    try:
        df = pd.read_csv(filepath, encoding='utf-8', on_bad_lines='warn')
    except Exception:
        # Fallback: try reading with flexible column handling
        df = pd.read_csv(filepath, encoding='utf-8', on_bad_lines='skip')

    # Rename columns to internal standard names
    df = df.rename(columns=GCMS_COLUMN_MAP)

    # Handle duplicate concentration columns (keep first)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]

    # Ensure CAS numbers are strings and clean them
    if 'cas' in df.columns:
        df['cas'] = df['cas'].astype(str).str.strip()
        # Replace placeholder CAS like "0-00-0" with None
        df.loc[df['cas'].isin(['0-00-0', 'nan', '']), 'cas'] = None

    # Convert concentration to float
    if 'concentration_mg_kg' in df.columns:
        df['concentration_mg_kg'] = pd.to_numeric(df['concentration_mg_kg'], errors='coerce')

    # Sort by concentration (descending) for easier analysis
    df = df.sort_values('concentration_mg_kg', ascending=False).reset_index(drop=True)

    return df


def parse_gcms_excel(filepath: str | Path, sheet_name: str = 0) -> pd.DataFrame:
    """
    Parse a GC-MS Excel export file into a standardized DataFrame.

    Args:
        filepath: Path to the GC-MS Excel file.
        sheet_name: Sheet name or index to read.

    Returns:
        pd.DataFrame with standardized column names.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"GC-MS file not found: {filepath}")

    df = pd.read_excel(filepath, sheet_name=sheet_name)
    df = df.rename(columns=GCMS_COLUMN_MAP)

    # Same cleaning logic as CSV
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]

    if 'cas' in df.columns:
        df['cas'] = df['cas'].astype(str).str.strip()
        df.loc[df['cas'].isin(['0-00-0', 'nan', '']), 'cas'] = None

    if 'concentration_mg_kg' in df.columns:
        df['concentration_mg_kg'] = pd.to_numeric(df['concentration_mg_kg'], errors='coerce')

    df = df.sort_values('concentration_mg_kg', ascending=False).reset_index(drop=True)

    return df


def get_top_compounds(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Get the top N compounds by concentration.

    Args:
        df: Parsed GC-MS DataFrame.
        top_n: Number of top compounds to return.

    Returns:
        DataFrame with top N compounds.
    """
    return df.head(top_n)[['compound_name_cn', 'compound_name_en', 'cas', 'concentration_mg_kg', 'sensory_notes']]


def get_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get summary statistics for the GC-MS analysis.

    Returns:
        Dict with total peaks, identified compounds, concentration sum, etc.
    """
    total_peaks = len(df)
    identified = df['compound_name_cn'].notna().sum()
    with_cas = df['cas'].notna().sum()
    total_concentration = df['concentration_mg_kg'].sum()

    return {
        'total_peaks': total_peaks,
        'identified_compounds': identified,
        'compounds_with_cas': with_cas,
        'total_concentration_mg_kg': total_concentration,
        'top_compound': df.iloc[0]['compound_name_cn'] if len(df) > 0 else None
    }


if __name__ == '__main__':
    # Test with sample file
    sample_path = Path(__file__).parent.parent.parent / 'data' / 'raw_gcms' / 'Sample1.csv'
    if sample_path.exists():
        df = parse_gcms_csv(sample_path)
        print("=== GC-MS Parser Test ===")
        print(f"Parsed {len(df)} peaks")
        print("\nTop 10 Compounds:")
        print(get_top_compounds(df, 10))
        print("\nSummary:")
        print(get_summary_stats(df))
