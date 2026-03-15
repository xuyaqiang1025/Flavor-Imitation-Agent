# Flavor Imitation Agent - Knowledge Module Initialization
"""
src.knowledge - Knowledge Base Module
=====================================
Manages raw material inventory, sensory mappings, and chemical databases.
"""
from .inventory_manager import InventoryManager, RawMaterial, MaterialType, generate_sample_inventory

__all__ = [
    'InventoryManager',
    'RawMaterial',
    'MaterialType',
    'generate_sample_inventory'
]
