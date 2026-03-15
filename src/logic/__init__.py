# Flavor Imitation Agent - Logic Module Initialization
"""
src.logic - Formula Logic & Validation
========================================
Contains sensory validation and feedback translation logic.
"""
from .sensory_validator import SensoryValidator, ValidationIssue
from .translator import SensoryTranslator, FormulaAdjustment

__all__ = [
    'SensoryValidator',
    'ValidationIssue',
    'SensoryTranslator',
    'FormulaAdjustment'
]
