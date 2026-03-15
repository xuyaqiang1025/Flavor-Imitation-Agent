"""
Flavor Imitation Agent - Deconvolution Engine
===============================================
Core algorithm for separating Natural Extracts from Synthetic Monomers.
Implements the "Iterative Subtraction" method with LLM-assisted inference.
"""
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import sys
sys.path.insert(0, str(__file__).rsplit('src', 1)[0] + 'src')

from knowledge.natural_fingerprints import NaturalFingerprintDB, NaturalFingerprint
from knowledge.inventory_manager import InventoryManager


@dataclass
class DeconvolutionResult:
    """Result of the deconvolution analysis."""
    # Identified natural extracts: name -> estimated percentage
    naturals: Dict[str, float] = field(default_factory=dict)
    # Residual synthetic monomers: CAS -> concentration
    synthetics: Dict[str, float] = field(default_factory=dict)
    # Confidence scores
    confidence: float = 0.0
    # Reasoning trace for explainability
    reasoning: List[str] = field(default_factory=list)
    # Warnings/uncertainties
    warnings: List[str] = field(default_factory=list)


class DeconvolutionEngine:
    """
    Main engine for reverse-engineering flavor formulas.
    Uses iterative subtraction to separate naturals from synthetics.
    """

    def __init__(
        self,
        fingerprint_db: Optional[NaturalFingerprintDB] = None,
        inventory_mgr: Optional[InventoryManager] = None
    ):
        self.fp_db = fingerprint_db or NaturalFingerprintDB()
        self.inventory = inventory_mgr

    def analyze(
        self,
        gcms_df: pd.DataFrame,
        max_naturals: int = 5,
        min_match_score: float = 30.0
    ) -> DeconvolutionResult:
        """
        Analyze GC-MS data to deconvolute naturals from synthetics.

        Args:
            gcms_df: Cleaned GC-MS DataFrame with 'cas' and 'concentration_mg_kg' columns.
            max_naturals: Maximum number of natural extracts to identify.
            min_match_score: Minimum score to consider a natural as candidate.

        Returns:
            DeconvolutionResult with identified components.
        """
        result = DeconvolutionResult()

        if 'cas' not in gcms_df.columns:
            result.warnings.append("No CAS column found in data")
            return result

        # Step 1: Get all detected CAS numbers
        detected_cas = gcms_df['cas'].dropna().tolist()
        cas_to_conc = dict(zip(gcms_df['cas'], gcms_df['concentration_mg_kg']))
        total_conc = gcms_df['concentration_mg_kg'].sum()

        result.reasoning.append(f"检测到 {len(detected_cas)} 个化合物，总浓度 {total_conc:.0f} mg/kg")

        # Step 2: Find natural extract candidates
        candidates = self.fp_db.find_candidates(detected_cas)
        result.reasoning.append(f"天然提取物候选: {len(candidates)} 个")

        # Step 3: Iterative subtraction
        remaining_cas = set(detected_cas)
        remaining_conc = cas_to_conc.copy()
        identified_naturals = []

        for nat_id, score in candidates.items():
            if score < min_match_score:
                continue
            if len(identified_naturals) >= max_naturals:
                break

            fp = self.fp_db.get_by_id(nat_id)
            if not fp:
                continue

            # Estimate natural's contribution
            estimate = self._estimate_natural_amount(fp, remaining_conc)
            if estimate['percentage'] > 0.5:  # At least 0.5% contribution
                identified_naturals.append((nat_id, fp, estimate))
                result.reasoning.append(
                    f"识别到 {fp.name_cn}: 约 {estimate['percentage']:.1f}% (基于 {estimate['basis']})"
                )

                # Subtract this natural's contribution
                for cas, pct in fp.composition.items():
                    if cas in remaining_conc:
                        subtract_amount = estimate['amount_mg_kg'] * (pct / 100)
                        remaining_conc[cas] = max(0, remaining_conc[cas] - subtract_amount)
                        if remaining_conc[cas] < 1:  # Below detection threshold
                            remaining_cas.discard(cas)

        # Step 4: Compile results
        for nat_id, fp, estimate in identified_naturals:
            result.naturals[fp.name_cn] = estimate['percentage']

        # Step 5: Remaining peaks are assumed synthetics
        for cas, conc in remaining_conc.items():
            if conc > 5:  # Only significant residuals
                result.synthetics[cas] = conc

        # Step 6: Calculate confidence
        explained_pct = sum(result.naturals.values())
        synthetic_pct = (sum(result.synthetics.values()) / total_conc * 100) if total_conc > 0 else 0
        result.confidence = min(100, explained_pct + synthetic_pct)

        # Add warnings for low confidence
        if result.confidence < 50:
            result.warnings.append("警告: 解析置信度较低，可能存在未知天然物或配方")
        if len(result.naturals) == 0:
            result.warnings.append("未识别到明显的天然提取物，配方可能为纯合成")

        return result

    def _estimate_natural_amount(
        self,
        fp: NaturalFingerprint,
        cas_to_conc: Dict[str, float]
    ) -> Dict:
        """
        Estimate the amount of a natural extract based on detected peaks.
        Uses marker molecules when available, otherwise uses major components.
        """
        estimates = []
        basis = "major components"

        # Priority 1: Use marker molecules
        for marker_cas in fp.markers:
            if marker_cas in cas_to_conc and cas_to_conc[marker_cas] > 0:
                marker_pct_in_natural = fp.composition.get(marker_cas, 0)
                if marker_pct_in_natural > 0:
                    # Back-calculate: if marker is X% of natural, and we detected Y mg/kg of marker
                    estimated_natural_conc = cas_to_conc[marker_cas] / (marker_pct_in_natural / 100)
                    estimates.append(estimated_natural_conc)
                    basis = f"marker: {marker_cas}"

        # Priority 2: Use major components (> 20%)
        if not estimates:
            major_components = fp.get_major_components(threshold=20.0)
            for cas, pct in major_components.items():
                if cas in cas_to_conc and cas_to_conc[cas] > 0:
                    estimated_natural_conc = cas_to_conc[cas] / (pct / 100)
                    estimates.append(estimated_natural_conc)

        if not estimates:
            return {'amount_mg_kg': 0, 'percentage': 0, 'basis': 'no match'}

        # Take median estimate to reduce outlier impact
        estimates.sort()
        median_estimate = estimates[len(estimates) // 2]

        # Convert to percentage (assuming 1,000,000 mg/kg = 100%)
        total_flavor_base = sum(cas_to_conc.values())
        percentage = (median_estimate / total_flavor_base * 100) if total_flavor_base > 0 else 0

        return {
            'amount_mg_kg': median_estimate,
            'percentage': percentage,
            'basis': basis
        }

    def generate_formula(
        self,
        result: DeconvolutionResult,
        gcms_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate a structured formula from the deconvolution result.

        Returns:
            DataFrame with columns: [Name, Type, Percentage, Role, Notes]
        """
        formula_rows = []

        # Add naturals
        for name, pct in result.naturals.items():
            formula_rows.append({
                'Name': name,
                'Type': '天然',
                'Percentage': round(pct, 2),
                'Role': '复合香基',
                'Notes': ''
            })

        # Add synthetics (lookup names from inventory or GC-MS data)
        for cas, conc in result.synthetics.items():
            name = cas  # Default to CAS
            role = '修饰'

            # Try to get name from GC-MS data
            match = gcms_df[gcms_df['cas'] == cas]
            if len(match) > 0:
                if 'compound_name_cn' in match.columns and pd.notna(match.iloc[0]['compound_name_cn']):
                    name = match.iloc[0]['compound_name_cn']
                if 'sensory_notes' in match.columns and pd.notna(match.iloc[0]['sensory_notes']):
                    notes = match.iloc[0]['sensory_notes']
                else:
                    notes = ''
            else:
                notes = ''

            # Calculate percentage
            total_conc = gcms_df['concentration_mg_kg'].sum()
            pct = (conc / total_conc * 100) if total_conc > 0 else 0

            formula_rows.append({
                'Name': name,
                'Type': '合成',
                'Percentage': round(pct, 2),
                'Role': role,
                'Notes': notes if 'notes' in dir() else ''
            })

        # Sort by percentage descending
        formula_df = pd.DataFrame(formula_rows)
        if len(formula_df) > 0:
            formula_df = formula_df.sort_values('Percentage', ascending=False).reset_index(drop=True)

        return formula_df


if __name__ == '__main__':
    # Test the deconvolution engine
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from ingestion.parser import parse_gcms_csv
    from ingestion.cleaner import clean_gcms_data, merge_duplicate_compounds

    # Load test data
    sample_path = Path(__file__).parent.parent.parent / 'data' / 'raw_gcms' / 'Sample1.csv'
    if sample_path.exists():
        df = parse_gcms_csv(sample_path)
        df_clean = clean_gcms_data(df)
        df_merged = merge_duplicate_compounds(df_clean)

        print("=== Deconvolution Engine Test ===")
        print(f"Input: {len(df_merged)} compounds")

        engine = DeconvolutionEngine()
        result = engine.analyze(df_merged)

        print("\n--- Reasoning Trace ---")
        for r in result.reasoning:
            print(f"  {r}")

        print("\n--- Identified Naturals ---")
        for name, pct in result.naturals.items():
            print(f"  {name}: {pct:.1f}%")

        print(f"\n--- Residual Synthetics: {len(result.synthetics)} compounds ---")

        print(f"\nConfidence: {result.confidence:.1f}%")
        if result.warnings:
            print("\nWarnings:")
            for w in result.warnings:
                print(f"  ⚠ {w}")

        # Generate formula
        formula = engine.generate_formula(result, df_merged)
        print("\n--- Generated Formula ---")
        print(formula.head(15).to_string(index=False))
