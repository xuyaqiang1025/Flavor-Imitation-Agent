# Flavor Imitation Agent - Python Package Initialization
"""
src.ingestion - Data Ingestion Module
=====================================
Handles reading and cleaning GC-MS data files.
"""
from .parser import parse_gcms_csv, parse_gcms_excel, get_top_compounds, get_summary_stats
from .cleaner import clean_gcms_data, predict_acetals, merge_duplicate_compounds

__all__ = [
    'parse_gcms_csv',
    'parse_gcms_excel',
    'get_top_compounds',
    'get_summary_stats',
    'clean_gcms_data',
    'predict_acetals',
    'merge_duplicate_compounds'
]
