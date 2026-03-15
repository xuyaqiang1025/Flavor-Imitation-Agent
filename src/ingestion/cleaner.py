"""
Flavor Imitation Agent - Data Cleaner
======================================
Filters out noise, reaction byproducts, and non-flavor compounds from GC-MS data.
Implements the "Virtual Reactor" logic to predict and exclude acetals/artifacts.
"""
import pandas as pd
from typing import Set, List, Dict
import re


# ==================== BLOCKLISTS ====================
# Common solvents and carriers (should be excluded from flavor analysis)
SOLVENT_CAS = {
    '57-55-6',   # Propylene Glycol (PG)
    '56-81-5',   # Glycerol (VG)
    '64-17-5',   # Ethanol
    '67-56-1',   # Methanol
    '67-64-1',   # Acetone
    '75-05-8',   # Acetonitrile
}

# Column bleed / Siloxane artifacts (common GC artifacts)
SILOXANE_CAS = {
    '556-67-2',   # Octamethylcyclotetrasiloxane (D4)
    '541-05-9',   # Hexamethylcyclotrisiloxane (D3)
    '540-97-6',   # Dodecamethylcyclohexasiloxane (D6)
    '541-02-6',   # Decamethylcyclopentasiloxane (D5)
}

# Nicotine and related (excluded from flavor calculation)
NICOTINE_CAS = {
    '54-11-5',    # Nicotine
    '22083-74-5', # Nicotine salt
}

# Known reaction products (Acetals formed by Aldehydes + PG/VG)
KNOWN_ACETALS = {
    '68527-74-2',  # Vanillin propylene glycol acetal
    # More can be added as we learn about them
}

# Plasticizers / Contaminants
CONTAMINANT_CAS = {
    '84-69-5',    # Diisobutyl phthalate (DIBP)
    '117-81-7',   # Bis(2-ethylhexyl) phthalate (DEHP)
    '84-74-2',    # Dibutyl phthalate (DBP)
}


# ==================== VIRTUAL REACTOR ====================
# Aldehydes that commonly form acetals with PG
ALDEHYDE_CAS_TO_ACETAL = {
    '121-33-5': '68527-74-2',  # Vanillin -> Vanillin PG Acetal
    '121-32-4': None,          # Ethyl Vanillin -> Ethyl Vanillin PG Acetal (CAS TBD)
    # Add more aldehyde -> acetal mappings as needed
}


def get_all_blocklist_cas() -> Set[str]:
    """Get the full set of CAS numbers to exclude."""
    return SOLVENT_CAS | SILOXANE_CAS | NICOTINE_CAS | KNOWN_ACETALS | CONTAMINANT_CAS


def clean_gcms_data(
    df: pd.DataFrame,
    remove_solvents: bool = True,
    remove_nicotine: bool = True,
    remove_siloxanes: bool = True,
    remove_acetals: bool = True,
    remove_contaminants: bool = True,
    min_concentration: float = 1.0,
    min_match_factor: float = 80.0
) -> pd.DataFrame:
    """
    Clean GC-MS data by removing noise and unwanted compounds.

    Args:
        df: Raw parsed GC-MS DataFrame.
        remove_solvents: Remove PG/VG/Ethanol etc.
        remove_nicotine: Remove nicotine peaks.
        remove_siloxanes: Remove column bleed artifacts.
        remove_acetals: Remove known reaction products.
        remove_contaminants: Remove plasticizers.
        min_concentration: Minimum concentration threshold (mg/kg).
        min_match_factor: Minimum match factor for reliable identification.

    Returns:
        Cleaned DataFrame with noise removed.
    """
    df_clean = df.copy()
    original_count = len(df_clean)

    # Build blocklist based on settings
    blocklist = set()
    if remove_solvents:
        blocklist |= SOLVENT_CAS
    if remove_nicotine:
        blocklist |= NICOTINE_CAS
    if remove_siloxanes:
        blocklist |= SILOXANE_CAS
    if remove_acetals:
        blocklist |= KNOWN_ACETALS
    if remove_contaminants:
        blocklist |= CONTAMINANT_CAS

    # Filter by CAS blocklist
    if 'cas' in df_clean.columns and blocklist:
        df_clean = df_clean[~df_clean['cas'].isin(blocklist)]

    # Filter by concentration threshold
    if 'concentration_mg_kg' in df_clean.columns:
        df_clean = df_clean[df_clean['concentration_mg_kg'] >= min_concentration]

    # Filter by match factor (only for identified compounds)
    if 'match_factor' in df_clean.columns:
        # Keep rows where match_factor is NaN (unidentified) OR above threshold
        df_clean = df_clean[
            (df_clean['match_factor'].isna()) | 
            (df_clean['match_factor'] >= min_match_factor)
        ]

    # Remove compounds with "#N/A" or empty Chinese names (unidentified)
    if 'compound_name_cn' in df_clean.columns:
        df_clean = df_clean[
            (df_clean['compound_name_cn'].notna()) &
            (df_clean['compound_name_cn'] != '#N/A')
        ]

    removed_count = original_count - len(df_clean)
    print(f"[Cleaner] Removed {removed_count} peaks ({original_count} -> {len(df_clean)})")

    return df_clean.reset_index(drop=True)


def predict_acetals(df: pd.DataFrame) -> List[Dict]:
    """
    Predict potential acetals based on detected aldehydes.
    Uses the "Virtual Reactor" logic: Aldehyde + PG -> Acetal.

    Args:
        df: GC-MS DataFrame (pre or post cleaning).

    Returns:
        List of predicted acetal compounds likely to be artifacts.
    """
    predictions = []

    if 'cas' not in df.columns:
        return predictions

    detected_cas = set(df['cas'].dropna().tolist())

    for aldehyde_cas, acetal_cas in ALDEHYDE_CAS_TO_ACETAL.items():
        if aldehyde_cas in detected_cas:
            aldehyde_name = df[df['cas'] == aldehyde_cas]['compound_name_cn'].iloc[0] if len(df[df['cas'] == aldehyde_cas]) > 0 else "Unknown"
            predictions.append({
                'parent_aldehyde_cas': aldehyde_cas,
                'parent_aldehyde_name': aldehyde_name,
                'predicted_acetal_cas': acetal_cas,
                'reason': f"{aldehyde_name} + PG -> Acetal (Reaction Product)"
            })

    return predictions


def merge_duplicate_compounds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge rows with the same CAS number (same compound appearing at different RT).
    Sum up their concentrations.

    Args:
        df: Cleaned GC-MS DataFrame.

    Returns:
        DataFrame with duplicates merged.
    """
    if 'cas' not in df.columns or 'concentration_mg_kg' not in df.columns:
        return df

    # For compounds with CAS, group and sum
    has_cas = df[df['cas'].notna()].copy()
    no_cas = df[df['cas'].isna()].copy()

    if len(has_cas) > 0:
        # Aggregate: sum concentration, keep first row's other data
        aggregated = has_cas.groupby('cas').agg({
            'compound_name_cn': 'first',
            'compound_name_en': 'first',
            'sensory_notes': 'first',
            'fema_id': 'first',
            'molecular_formula': 'first',
            'concentration_mg_kg': 'sum',
            'retention_time': 'first',
            'match_factor': 'max'
        }).reset_index()

        # Combine back with no-CAS rows
        result = pd.concat([aggregated, no_cas], ignore_index=True)
    else:
        result = df

    return result.sort_values('concentration_mg_kg', ascending=False).reset_index(drop=True)


if __name__ == '__main__':
    # Test with parser
    from parser import parse_gcms_csv
    from pathlib import Path

    sample_path = Path(__file__).parent.parent.parent / 'data' / 'raw_gcms' / 'Sample1.csv'
    if sample_path.exists():
        raw_df = parse_gcms_csv(sample_path)
        print(f"Raw data: {len(raw_df)} peaks")

        clean_df = clean_gcms_data(raw_df)
        print(f"Clean data: {len(clean_df)} peaks")

        merged_df = merge_duplicate_compounds(clean_df)
        print(f"After merging duplicates: {len(merged_df)} compounds")

        # Check for predicted acetals
        predictions = predict_acetals(raw_df)
        if predictions:
            print("\nPredicted Acetals (Reaction Products):")
            for p in predictions:
                print(f"  - {p['reason']}")
