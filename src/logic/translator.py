"""
Flavor Imitation Agent - Sensory to Chemical Translator
=========================================================
Translates user sensory feedback into chemical formula adjustments.
E.g., "Too sweet" -> "Reduce Ethyl Maltol by 20%"
"""
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import re


@dataclass
class FormulaAdjustment:
    """A suggested adjustment to the formula."""
    action: str             # "increase", "decrease", "add", "remove"
    target: str             # Ingredient name or CAS
    amount: Optional[float] # Percentage change (e.g., 0.5 for +0.5%)
    reason: str             # Explanation


# Sensory keyword to chemical action mapping
SENSORY_MAPPINGS = {
    # Sweetness
    '甜': {
        'increase': ['乙基麦芽酚', '麦芽酚', '香兰素', 'Ethyl Maltol', 'Maltol', 'Vanillin'],
        'decrease': ['苦味物质', 'Quinoline'],
        'default_amount': 0.5
    },
    '太甜': {
        'decrease': ['乙基麦芽酚', '麦芽酚', 'Ethyl Maltol', 'Maltol'],
        'default_amount': -0.3
    },
    '不够甜': {
        'increase': ['乙基麦芽酚', '麦芽酚', 'Ethyl Maltol', 'Maltol'],
        'default_amount': 0.5
    },

    # Cooling
    '凉': {
        'increase': ['WS-23', '薄荷脑', 'Menthol', 'WS-3'],
        'default_amount': 0.3
    },
    '太凉': {
        'decrease': ['WS-23', '薄荷脑', 'Menthol', 'WS-3'],
        'default_amount': -0.2
    },
    '不够凉': {
        'increase': ['WS-23', 'WS-3'],
        'default_amount': 0.5
    },

    # Vanilla/Cream
    '香草': {
        'increase': ['香兰素', '乙基香兰素', 'Vanillin', 'Ethyl Vanillin'],
        'default_amount': 0.3
    },
    '奶油': {
        'increase': ['乙基香兰素', '双乙酰', 'Diacetyl', 'Acetoin'],
        'default_amount': 0.2
    },

    # Fruit
    '果味': {
        'increase': ['酯类', 'γ-癸内酯', 'gamma-Decalactone'],
        'default_amount': 0.3
    },
    '桃子': {
        'increase': ['γ-癸内酯', '桃醛', 'gamma-Decalactone', 'gamma-Undecalactone'],
        'default_amount': 0.2
    },

    # Tobacco
    '烟草': {
        'increase': ['吡嗪类', '喹啉', 'Quinoline', 'Trimethylpyrazine'],
        'default_amount': 0.1
    },
    '烟熏': {
        'increase': ['愈创木酚', 'Guaiacol', '丁香酚'],
        'default_amount': 0.05
    },

    # Negative descriptors
    '刺激': {
        'decrease': ['醛类', '酸类', 'Aldehyde'],
        'default_amount': -0.2
    },
    '化学感': {
        'decrease': ['高浓度单体', '醛类'],
        'action': 'review',
        'default_amount': -0.1
    },
    '苦': {
        'decrease': ['吡嗪类', '喹啉', 'Quinoline'],
        'default_amount': -0.05
    },
    '杂气': {
        'action': 'review',
        'note': '检查是否有反应副产物或过期原料'
    },
}


class SensoryTranslator:
    """
    Translates sensory feedback into formula adjustments.
    """

    def __init__(self, current_formula: Optional[Dict[str, float]] = None):
        """
        Args:
            current_formula: Dict of ingredient_name -> percentage
        """
        self.formula = current_formula or {}

    def translate(self, feedback: str) -> List[FormulaAdjustment]:
        """
        Translate a sensory feedback string into formula adjustments.

        Args:
            feedback: User's sensory description, e.g., "太甜，不够凉"

        Returns:
            List of FormulaAdjustment suggestions.
        """
        adjustments = []
        feedback_lower = feedback.lower()

        for keyword, mapping in SENSORY_MAPPINGS.items():
            if keyword in feedback or keyword.lower() in feedback_lower:
                # Check for increase actions
                if 'increase' in mapping:
                    for target in mapping['increase']:
                        # Check if target is in current formula
                        in_formula = any(target in ing for ing in self.formula.keys())
                        if in_formula or len(self.formula) == 0:
                            adjustments.append(FormulaAdjustment(
                                action='increase',
                                target=target,
                                amount=mapping.get('default_amount', 0.3),
                                reason=f"基于'{keyword}'的反馈，增加{target}"
                            ))
                            break  # Only suggest one target per keyword

                # Check for decrease actions
                if 'decrease' in mapping:
                    for target in mapping['decrease']:
                        in_formula = any(target in ing for ing in self.formula.keys())
                        if in_formula or len(self.formula) == 0:
                            adjustments.append(FormulaAdjustment(
                                action='decrease',
                                target=target,
                                amount=abs(mapping.get('default_amount', 0.3)),
                                reason=f"基于'{keyword}'的反馈，减少{target}"
                            ))
                            break

                # Check for review actions (needs human attention)
                if mapping.get('action') == 'review':
                    adjustments.append(FormulaAdjustment(
                        action='review',
                        target='FORMULA',
                        amount=None,
                        reason=f"'{keyword}'需要人工检查: {mapping.get('note', '请审核配方')}"
                    ))

        return adjustments

    def apply_adjustments(
        self,
        formula_df,
        adjustments: List[FormulaAdjustment]
    ):
        """
        Apply adjustments to a formula DataFrame.
        Returns a new DataFrame with modified percentages.
        """
        import pandas as pd
        new_formula = formula_df.copy()

        for adj in adjustments:
            if adj.action in ['increase', 'decrease'] and adj.amount:
                # Find matching ingredient
                mask = new_formula['Name'].astype(str).str.contains(adj.target, case=False, na=False)
                if mask.any():
                    if adj.action == 'increase':
                        new_formula.loc[mask, 'Percentage'] += adj.amount
                    else:
                        new_formula.loc[mask, 'Percentage'] -= adj.amount
                        new_formula.loc[mask, 'Percentage'] = new_formula.loc[mask, 'Percentage'].clip(lower=0)

        return new_formula

    def explain_adjustment(self, adjustment: FormulaAdjustment) -> str:
        """Generate a human-readable explanation of an adjustment."""
        if adjustment.action == 'increase':
            return f"↑ 增加 {adjustment.target} {adjustment.amount}% - {adjustment.reason}"
        elif adjustment.action == 'decrease':
            return f"↓ 减少 {adjustment.target} {adjustment.amount}% - {adjustment.reason}"
        elif adjustment.action == 'review':
            return f"⚠ 需要审核: {adjustment.reason}"
        else:
            return f"{adjustment.action}: {adjustment.target}"


if __name__ == '__main__':
    # Test the translator
    translator = SensoryTranslator({'乙基麦芽酚': 15.0, 'WS-23': 10.0, '香兰素': 5.0})

    test_feedbacks = [
        "太甜了，需要更凉一点",
        "香草味不够突出",
        "有点化学感，刺激",
    ]

    print("=== Sensory Translator Test ===\n")
    for feedback in test_feedbacks:
        print(f"用户反馈: '{feedback}'")
        adjustments = translator.translate(feedback)
        for adj in adjustments:
            print(f"  {translator.explain_adjustment(adj)}")
        print()
