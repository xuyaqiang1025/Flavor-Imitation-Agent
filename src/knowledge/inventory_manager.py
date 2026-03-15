"""
Flavor Imitation Agent - Raw Material Inventory Manager
=========================================================
Manages the user's available ingredient database.
Handles both synthetic monomers and natural extracts.
"""
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum


class MaterialType(Enum):
    SYNTHETIC = "合成"     # Synthetic monomer
    NATURAL = "天然"       # Natural extract
    UNKNOWN = "未知"


@dataclass
class RawMaterial:
    """Data class representing a single raw material."""
    id: str
    cas: Optional[str]
    name_cn: str
    name_en: Optional[str]
    molecular_formula: Optional[str]
    molecular_weight: Optional[float]
    boiling_point: Optional[float]
    solubility: Optional[str]
    crystal_form: Optional[str]
    max_usage_intl: Optional[float]   # 电子烟国际最大用量
    is_natural: bool
    usage_scope: Optional[str]        # 适用范围
    odor_description: Optional[str]   # 香气气味描述
    odor_profile: Optional[str]       # 香气气调描述
    tank_capacity: Optional[float]    # 塔容量(1% in PG)
    vape_evaluation: Optional[str]    # 抽吸评判描述
    cost_per_gram: Optional[float]    # 成本(元/g)


# Column mapping for raw material database (Chinese -> Internal)
INVENTORY_COLUMN_MAP = {
    '编号': 'id',
    'CAS': 'cas',
    '原料名称': 'name_cn',
    '英文名': 'name_en',
    '分子式': 'molecular_formula',
    '化学结构式': 'structure',  # May not be parseable
    '分子量': 'molecular_weight',
    '沸点': 'boiling_point',
    '溶解性': 'solubility',
    '晶态': 'crystal_form',
    '电子烟国际最大用量': 'max_usage_intl',
    '天然存在': 'is_natural',
    '适用范围': 'usage_scope',
    '香气气味描述': 'odor_description',
    '香气气调描述': 'odor_profile',
    '塔容量(1% in PG)': 'tank_capacity',
    '抽吸评判描述': 'vape_evaluation',
    '成本(元/g)': 'cost_per_gram'
}


class InventoryManager:
    """
    Manages the raw material inventory database.
    Provides lookup, search, and matching capabilities.
    """

    def __init__(self, inventory_path: Optional[str | Path] = None):
        self.inventory_df: Optional[pd.DataFrame] = None
        self._cas_index: Dict[str, int] = {}  # CAS -> row index for fast lookup

        if inventory_path:
            self.load_inventory(inventory_path)

    def load_inventory(self, filepath: str | Path) -> None:
        """
        Load inventory from CSV or Excel file.

        Args:
            filepath: Path to the inventory file.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Inventory file not found: {filepath}")

        # Determine file type and read
        if filepath.suffix.lower() == '.csv':
            self.inventory_df = pd.read_csv(filepath, encoding='utf-8')
        elif filepath.suffix.lower() in ['.xlsx', '.xls']:
            self.inventory_df = pd.read_excel(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")

        # Rename columns
        self.inventory_df = self.inventory_df.rename(columns=INVENTORY_COLUMN_MAP)

        # Clean CAS numbers
        if 'cas' in self.inventory_df.columns:
            self.inventory_df['cas'] = self.inventory_df['cas'].astype(str).str.strip()
            self.inventory_df.loc[self.inventory_df['cas'].isin(['nan', '', 'None']), 'cas'] = None

        # Build CAS index
        self._build_cas_index()

        print(f"[Inventory] Loaded {len(self.inventory_df)} raw materials")

    def _build_cas_index(self) -> None:
        """Build a CAS -> row index for fast lookup."""
        self._cas_index = {}
        if self.inventory_df is None or 'cas' not in self.inventory_df.columns:
            return

        for idx, row in self.inventory_df.iterrows():
            if pd.notna(row['cas']):
                self._cas_index[row['cas']] = idx

    def lookup_by_cas(self, cas: str) -> Optional[Dict[str, Any]]:
        """
        Look up a material by CAS number.

        Args:
            cas: CAS Registry Number.

        Returns:
            Dict with material info, or None if not found.
        """
        if cas in self._cas_index:
            row = self.inventory_df.iloc[self._cas_index[cas]]
            return row.to_dict()
        return None

    def search_by_name(self, name: str, language: str = 'cn') -> pd.DataFrame:
        """
        Search materials by name (partial match).

        Args:
            name: Search term.
            language: 'cn' for Chinese, 'en' for English.

        Returns:
            DataFrame of matching materials.
        """
        if self.inventory_df is None:
            return pd.DataFrame()

        col = 'name_cn' if language == 'cn' else 'name_en'
        if col not in self.inventory_df.columns:
            return pd.DataFrame()

        mask = self.inventory_df[col].astype(str).str.contains(name, case=False, na=False)
        return self.inventory_df[mask]

    def get_by_sensory(self, keywords: List[str]) -> pd.DataFrame:
        """
        Find materials matching sensory keywords (e.g., "甜的", "果味").

        Args:
            keywords: List of sensory descriptors.

        Returns:
            DataFrame of materials with matching odor profiles.
        """
        if self.inventory_df is None:
            return pd.DataFrame()

        if 'odor_description' not in self.inventory_df.columns:
            return pd.DataFrame()

        mask = pd.Series([False] * len(self.inventory_df))
        for kw in keywords:
            mask |= self.inventory_df['odor_description'].astype(str).str.contains(kw, case=False, na=False)

        return self.inventory_df[mask]

    def match_gcms_to_inventory(self, gcms_df: pd.DataFrame) -> pd.DataFrame:
        """
        Match GC-MS detected compounds to inventory.
        Adds 'in_inventory' and 'inventory_info' columns.

        Args:
            gcms_df: Cleaned GC-MS DataFrame with 'cas' column.

        Returns:
            DataFrame with inventory match information added.
        """
        result = gcms_df.copy()
        result['in_inventory'] = False
        result['inventory_match'] = None

        if 'cas' not in result.columns or self.inventory_df is None:
            return result

        for idx, row in result.iterrows():
            if pd.notna(row['cas']):
                inv_match = self.lookup_by_cas(row['cas'])
                if inv_match:
                    result.at[idx, 'in_inventory'] = True
                    result.at[idx, 'inventory_match'] = inv_match.get('name_cn', row['cas'])

        matched = result['in_inventory'].sum()
        print(f"[Inventory Match] {matched}/{len(result)} compounds found in inventory")

        return result

    def get_natural_materials(self) -> pd.DataFrame:
        """Get all natural extracts from inventory."""
        if self.inventory_df is None:
            return pd.DataFrame()

        if 'is_natural' not in self.inventory_df.columns:
            return pd.DataFrame()

        # Check various ways "natural" might be indicated
        mask = (
            (self.inventory_df['is_natural'].astype(str).str.contains('是|天然|yes|true', case=False, na=False)) |
            (self.inventory_df['is_natural'] == True) |
            (self.inventory_df['is_natural'] == 1)
        )
        return self.inventory_df[mask]

    def get_vape_compatible(self, max_bp: float = 300) -> pd.DataFrame:
        """
        Get materials suitable for vaping (low enough boiling point).

        Args:
            max_bp: Maximum boiling point in Celsius.

        Returns:
            DataFrame of vape-compatible materials.
        """
        if self.inventory_df is None or 'boiling_point' not in self.inventory_df.columns:
            return pd.DataFrame()

        mask = self.inventory_df['boiling_point'] <= max_bp
        return self.inventory_df[mask]


def generate_sample_inventory() -> pd.DataFrame:
    """
    Generate a sample inventory dataset for testing.
    Based on common e-cigarette flavor ingredients.
    """
    data = [
        # ID, CAS, 原料名称, 英文名, 分子式, 化学结构式, 分子量, 沸点, 溶解性, 晶态, 电子烟国际最大用量, 天然存在, 适用范围, 香气气味描述, 香气气调描述, 塔容量(1% in PG), 抽吸评判描述, 成本(元/g)
        ('001', '4940-11-8', '乙基麦芽酚', 'Ethyl Maltol', 'C7H8O3', None, 140.14, 161, '溶于水、乙醇', '针状晶体', 5.0, '否', '食品/电子烟', '甜的，焦糖，果酱', '甜香', 8.5, '入口顺滑，回甜', 0.35),
        ('002', '121-33-5', '香兰素', 'Vanillin', 'C8H8O3', None, 152.15, 285, '微溶于水，溶于乙醇', '针状晶体', 10.0, '是', '食品/电子烟', '甜的，香草，奶油味', '香草调', 7.2, '香草味突出，后调持久', 0.42),
        ('003', '51115-67-4', 'WS-23', 'WS-23', 'C10H21NO', None, 171.28, 120, '溶于PG', '无色液体', 3.0, '否', '电子烟', '凉，薄荷醇', '凉感', 9.0, '清凉感强，无刺激', 1.20),
        ('004', '89-78-1', '薄荷脑', 'dl-Menthol', 'C10H20O', None, 156.27, 212, '微溶于水，溶于乙醇', '棱柱状晶体', 8.0, '是', '食品/电子烟', '胡椒薄荷，凉，木质', '薄荷调', 8.0, '凉感自然，略带辛辣', 0.28),
        ('005', '121-32-4', '乙基香兰素', 'Ethyl Vanillin', 'C9H10O3', None, 166.18, 295, '微溶于水，溶于乙醇', '针状晶体', 8.0, '否', '食品/电子烟', '甜的，奶油味，香草', '香草调', 7.5, '香草味更浓郁', 0.55),
        ('006', '706-14-9', 'γ-癸内酯', 'gamma-Decalactone', 'C10H18O2', None, 170.25, 281, '微溶于水，溶于乙醇', '无色液体', 5.0, '是', '食品/电子烟', '桃子，椰子，甜的', '果香调', 6.8, '桃味自然，甜润', 0.88),
        ('007', '104-67-6', '桃醛', 'gamma-Undecalactone', 'C11H20O2', None, 184.28, 297, '微溶于水', '无色液体', 4.0, '是', '食品/电子烟', '果味，桃子，奶油味', '果香调', 6.5, '桃味浓郁，后调甜', 0.95),
        ('008', '118-71-8', '麦芽酚', 'Maltol', 'C6H6O3', None, 126.11, 160, '溶于热水', '针状晶体', 6.0, '是', '食品/电子烟', '甜的，焦糖，棉花糖', '甜香', 7.8, '焦糖感明显', 0.30),
        ('009', '91-22-5', '喹啉', 'Quinoline', 'C9H7N', None, 129.16, 238, '微溶于水，溶于乙醇', '无色液体', 0.5, '是', '烟草', '药草，霉味，烟草', '烟草调', 3.2, '烟草本味，略苦', 0.65),
        ('010', '14667-55-1', '2,3,5-三甲基吡嗪', 'Trimethylpyrazine', 'C7H10N2', None, 122.17, 171, '溶于水和乙醇', '无色液体', 2.0, '是', '食品/电子烟', '坚果，可可，烤的', '坚果调', 5.5, '坚果味突出', 0.48),
        ('011', '22047-25-2', '2-乙酰基吡嗪', 'Acetylpyrazine', 'C6H6N2O', None, 122.13, 185, '溶于水', '白色晶体', 1.5, '否', '食品/电子烟', '爆米花，坚果，玉米', '坚果调', 6.0, '爆米花香明显', 0.52),
        ('012', '8007-08-7', '姜油', 'Ginger Oil', None, None, None, None, '溶于乙醇', '黄色液体', 2.0, '是', '食品/电子烟', '辛辣，姜，木质', '辛香调', 4.5, '姜味自然，微辛', 1.85),
        ('013', '8016-20-4', '柠檬油', 'Lemon Oil', None, None, None, None, '溶于乙醇', '淡黄色液体', 3.0, '是', '食品/电子烟', '柑橘，柠檬，新鲜', '柑橘调', 5.8, '柠檬味清新', 1.25),
        ('014', '8008-57-9', '甜橙油', 'Sweet Orange Oil', None, None, None, None, '溶于乙醇', '橙黄色液体', 3.5, '是', '食品/电子烟', '甜橙，果味，新鲜', '柑橘调', 6.2, '甜橙味饱满', 0.95),
        ('015', '79-77-6', 'β-紫罗酮', 'beta-Ionone', 'C13H20O', None, 192.30, 266, '微溶于水', '无色液体', 1.0, '是', '食品/电子烟', '花的，木质，浆果', '花香调', 4.8, '花香雅致', 1.10),
    ]

    columns = ['编号', 'CAS', '原料名称', '英文名', '分子式', '化学结构式', '分子量', '沸点', '溶解性', '晶态',
               '电子烟国际最大用量', '天然存在', '适用范围', '香气气味描述', '香气气调描述',
               '塔容量(1% in PG)', '抽吸评判描述', '成本(元/g)']

    return pd.DataFrame(data, columns=columns)


if __name__ == '__main__':
    # Generate and save sample inventory
    sample_df = generate_sample_inventory()
    output_path = Path(__file__).parent.parent.parent / 'data' / 'inventory' / 'sample_inventory.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Sample inventory saved to: {output_path}")

    # Test loading
    mgr = InventoryManager(output_path)
    print("\n=== Inventory Test ===")
    print(f"Total materials: {len(mgr.inventory_df)}")

    # Test lookup
    vanillin = mgr.lookup_by_cas('121-33-5')
    if vanillin:
        print(f"\nLookup Vanillin: {vanillin['name_cn']} - {vanillin['odor_description']}")

    # Test search
    sweet = mgr.get_by_sensory(['甜的'])
    print(f"\nMaterials with '甜的': {len(sweet)}")

    # Test natural materials
    naturals = mgr.get_natural_materials()
    print(f"Natural materials: {len(naturals)}")
