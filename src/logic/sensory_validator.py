"""
Flavor Imitation Agent - Sensory Validator
============================================
Validates formula for "Smell vs. Vape" consistency.
Checks physical properties to ensure ingredients will perform well under atomization.
"""
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ValidationIssue:
    """A single validation issue found in the formula."""
    severity: str           # "warning", "error"
    ingredient: str         # Ingredient name
    issue_type: str         # e.g., "high_bp", "low_solubility"
    message: str            # Human-readable message
    suggestion: Optional[str] = None


# Physical property thresholds for vape compatibility
VAPE_THRESHOLDS = {
    'max_boiling_point': 300,      # °C - above this may carbonize
    'min_boiling_point': 100,      # °C - below this may evaporate in storage
    'max_molecular_weight': 400,   # Da - very heavy molecules don't vaporize well
}

# Known problematic ingredient categories
PROBLEMATIC_CATEGORIES = {
    'lipids': ['脂肪酸', '油酸', '亚油酸', '棕榈酸', 'fatty acid', 'lipid'],
    'sugars': ['蔗糖', '葡萄糖', '果糖', 'sugar', 'sucrose'],
    'proteins': ['蛋白', 'protein', 'amino acid'],
}


class SensoryValidator:
    """
    Validates formula ingredients for vape compatibility.
    Checks physical properties and flags potential issues.
    """

    def __init__(self, inventory_df: Optional[pd.DataFrame] = None):
        self.inventory = inventory_df

    def validate_formula(
        self,
        formula_df: pd.DataFrame,
        gcms_df: Optional[pd.DataFrame] = None
    ) -> List[ValidationIssue]:
        """
        Validate all ingredients in a formula.

        Args:
            formula_df: Formula DataFrame with Name, Type, Percentage columns.
            gcms_df: Original GC-MS data (for additional properties if needed).

        Returns:
            List of ValidationIssue objects.
        """
        issues = []

        for _, row in formula_df.iterrows():
            name = row.get('Name', 'Unknown')
            pct = row.get('Percentage', 0)

            # Skip very minor ingredients
            if pct < 0.1:
                continue

            # Check for problematic categories by name
            name_lower = str(name).lower()
            for category, keywords in PROBLEMATIC_CATEGORIES.items():
                if any(kw in name_lower for kw in keywords):
                    issues.append(ValidationIssue(
                        severity="warning",
                        ingredient=name,
                        issue_type=f"problematic_{category}",
                        message=f"'{name}' 属于 {category} 类，可能在雾化时产生问题",
                        suggestion=f"考虑减少用量或寻找替代物"
                    ))

            # Check boiling point if we have inventory data
            if self.inventory is not None and 'boiling_point' in self.inventory.columns:
                inv_match = self.inventory[
                    self.inventory['name_cn'].astype(str).str.contains(name, case=False, na=False)
                ]
                if len(inv_match) > 0:
                    bp = inv_match.iloc[0].get('boiling_point')
                    if pd.notna(bp) and bp > VAPE_THRESHOLDS['max_boiling_point']:
                        issues.append(ValidationIssue(
                            severity="warning",
                            ingredient=name,
                            issue_type="high_bp",
                            message=f"'{name}' 沸点 {bp}°C 过高，可能积碳或出香效率低",
                            suggestion=f"考虑用更轻分子替代或降低用量"
                        ))

        # Check for overall formula balance
        if len(formula_df) > 0:
            total_pct = formula_df['Percentage'].sum()
            if total_pct < 90:
                issues.append(ValidationIssue(
                    severity="info",
                    ingredient="TOTAL",
                    issue_type="low_coverage",
                    message=f"配方总和仅 {total_pct:.1f}%，可能存在未识别成分",
                    suggestion="检查是否有天然物被遗漏"
                ))

        return issues

    def check_smell_vs_vape_consistency(
        self,
        formula_df: pd.DataFrame
    ) -> Dict[str, any]:
        """
        Analyze formula for potential smell vs. vape discrepancies.

        Returns:
            Dict with 'score' (0-100) and 'issues' list.
        """
        result = {
            'score': 100,
            'issues': [],
            'high_bp_count': 0,
            'low_bp_count': 0
        }

        # Count ingredients by volatility category
        for _, row in formula_df.iterrows():
            name = str(row.get('Name', ''))
            pct = row.get('Percentage', 0)

            # Simple heuristic: heavy molecules (酸, 醇, 内酯) tend to have higher BP
            if any(kw in name for kw in ['酸', 'acid', '脂肪', 'lipid']):
                result['high_bp_count'] += 1
                result['score'] -= 5 * pct  # Penalize by percentage

        result['score'] = max(0, min(100, result['score']))

        if result['high_bp_count'] > 3:
            result['issues'].append(
                f"配方中有 {result['high_bp_count']} 种高沸点物质，可能导致抽吸体验弱于嗅香"
            )

        return result


def suggest_replacement(
    ingredient_name: str,
    issue_type: str
) -> Optional[str]:
    """
    Suggest a replacement for a problematic ingredient.
    Uses a simple lookup table.
    """
    REPLACEMENTS = {
        # High BP vanillins -> lighter alternatives
        ('香兰素', 'high_bp'): '使用乙基香兰素的低用量，或增加WS-23提升整体穿透力',
        ('香草', 'high_bp'): '考虑用香兰素丙二醇缩醛增加溶解性',
        # Fatty acids
        ('酸', 'problematic_lipids'): '减少用量至0.1%以下，或使用对应酯类替代',
    }

    for (name_kw, issue), suggestion in REPLACEMENTS.items():
        if name_kw in ingredient_name and issue == issue_type:
            return suggestion

    return None


if __name__ == '__main__':
    # Test with a sample formula
    sample_formula = pd.DataFrame([
        {'Name': '乙基麦芽酚', 'Type': '合成', 'Percentage': 25.0},
        {'Name': '香兰素', 'Type': '合成', 'Percentage': 8.0},
        {'Name': '棕榈酸', 'Type': '合成', 'Percentage': 0.5},
        {'Name': 'WS-23', 'Type': '合成', 'Percentage': 15.0},
        {'Name': '薄荷脑', 'Type': '天然', 'Percentage': 3.0},
    ])

    validator = SensoryValidator()
    issues = validator.validate_formula(sample_formula)

    print("=== Sensory Validator Test ===")
    print(f"Found {len(issues)} issues:\n")
    for issue in issues:
        print(f"[{issue.severity.upper()}] {issue.ingredient}")
        print(f"  {issue.message}")
        if issue.suggestion:
            print(f"  → {issue.suggestion}")
        print()

    consistency = validator.check_smell_vs_vape_consistency(sample_formula)
    print(f"Smell vs Vape Score: {consistency['score']}/100")
