"""
Flavor Imitation Agent - Natural Extract Fingerprint Database
===============================================================
Stores known compositions of natural extracts (essential oils, absolutes, etc.)
Used for the "Iterative Subtraction" deconvolution algorithm.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json
from pathlib import Path


@dataclass
class NaturalFingerprint:
    """
    Represents the chemical composition of a natural extract.
    """
    name_cn: str                          # Chinese name (e.g., "柠檬油")
    name_en: str                          # English name (e.g., "Lemon Oil")
    cas: Optional[str]                    # CAS if available
    category: str                         # e.g., "Citrus", "Floral", "Spice"
    # Composition: CAS -> percentage (0-100)
    composition: Dict[str, float] = field(default_factory=dict)
    # Marker molecules: CAS numbers that uniquely identify this natural
    markers: List[str] = field(default_factory=list)
    # Sensory descriptors
    sensory_notes: List[str] = field(default_factory=list)

    def get_major_components(self, threshold: float = 5.0) -> Dict[str, float]:
        """Get components above a certain percentage threshold."""
        return {cas: pct for cas, pct in self.composition.items() if pct >= threshold}


# ==================== BUILT-IN FINGERPRINT DATABASE ====================
# Based on public literature (Leffingwell, ISO standards, typical compositions)
# These are APPROXIMATE values - actual compositions vary by source/batch

NATURAL_FINGERPRINTS = {
    # Citrus Oils
    "lemon_oil": NaturalFingerprint(
        name_cn="柠檬油",
        name_en="Lemon Oil",
        cas="8008-56-8",
        category="Citrus",
        composition={
            "5989-27-5": 65.0,   # d-Limonene (major)
            "123-35-3": 12.0,    # beta-Myrcene
            "99-87-6": 8.0,      # p-Cymene
            "5392-40-5": 3.5,    # Citral (Geranial + Neral)
            "127-91-3": 2.5,     # beta-Pinene
            "80-56-8": 2.0,      # alpha-Pinene
        },
        markers=["5392-40-5", "5989-27-5"],  # Citral + Limonene strongly suggests Lemon
        sensory_notes=["柑橘", "柠檬", "新鲜", "酸"]
    ),
    "sweet_orange_oil": NaturalFingerprint(
        name_cn="甜橙油",
        name_en="Sweet Orange Oil",
        cas="8008-57-9",
        category="Citrus",
        composition={
            "5989-27-5": 95.0,   # d-Limonene (dominant!)
            "123-35-3": 2.0,     # beta-Myrcene
            "104-93-8": 0.5,     # p-Cresyl methyl ether
            "60-12-8": 0.3,      # Phenylethyl alcohol
        },
        markers=["5989-27-5"],  # Almost pure Limonene
        sensory_notes=["甜橙", "柑橘", "甜的", "新鲜"]
    ),
    "grapefruit_oil": NaturalFingerprint(
        name_cn="葡萄柚油",
        name_en="Grapefruit Oil",
        cas="8016-20-4",
        category="Citrus",
        composition={
            "5989-27-5": 90.0,   # d-Limonene
            "123-35-3": 3.0,     # beta-Myrcene
            "4674-50-4": 0.8,    # Nootkatone (KEY MARKER!)
            "7212-44-4": 0.5,    # Nootkatol
        },
        markers=["4674-50-4"],  # Nootkatone is unique to grapefruit
        sensory_notes=["葡萄柚", "柑橘", "苦", "清新"]
    ),

    # Mint/Cooling
    "peppermint_oil": NaturalFingerprint(
        name_cn="胡椒薄荷油",
        name_en="Peppermint Oil",
        cas="8006-90-4",
        category="Mint",
        composition={
            "89-78-1": 45.0,     # Menthol (major)
            "2216-51-5": 25.0,   # L-Menthone
            "89-80-5": 5.0,      # (+)-Pulegone
            "7212-44-4": 4.0,    # Menthyl acetate
            "470-82-6": 3.0,     # 1,8-Cineole (Eucalyptol)
        },
        markers=["89-78-1", "2216-51-5"],  # Menthol + Menthone
        sensory_notes=["胡椒薄荷", "凉", "清新", "草药"]
    ),
    "spearmint_oil": NaturalFingerprint(
        name_cn="留兰香油",
        name_en="Spearmint Oil",
        cas="8008-79-5",
        category="Mint",
        composition={
            "6485-40-1": 60.0,   # L-Carvone (KEY - different from peppermint!)
            "5989-27-5": 15.0,   # Limonene
            "123-35-3": 3.0,     # Myrcene
            "470-82-6": 2.0,     # 1,8-Cineole
        },
        markers=["6485-40-1"],  # Carvone is the key differentiator
        sensory_notes=["留兰香", "甜薄荷", "清新"]
    ),

    # Vanilla/Creamy
    "vanilla_extract": NaturalFingerprint(
        name_cn="香草提取物",
        name_en="Vanilla Extract",
        cas="8024-06-4",
        category="Vanilla",
        composition={
            "121-33-5": 85.0,    # Vanillin (dominant)
            "121-32-4": 5.0,     # Ethyl Vanillin
            "498-02-2": 3.0,     # Acetovanillone
            "93-51-6": 2.0,      # Creosol
        },
        markers=["121-33-5"],
        sensory_notes=["香草", "甜的", "奶油味"]
    ),

    # Tobacco-related
    "tobacco_absolute": NaturalFingerprint(
        name_cn="烟草浸膏",
        name_en="Tobacco Absolute",
        cas="8037-19-2",
        category="Tobacco",
        composition={
            "54-11-5": 5.0,      # Nicotine
            "91-22-5": 3.0,      # Quinoline
            "110-43-0": 2.5,     # 2-Heptanone
            "14667-55-1": 2.0,   # Trimethylpyrazine
            "4180-23-8": 1.5,    # trans-Anethole
        },
        markers=["91-22-5", "14667-55-1"],  # Quinoline + Pyrazines
        sensory_notes=["烟草", "干草", "木质", "坚果"]
    ),

    # Fruit
    "strawberry_essence": NaturalFingerprint(
        name_cn="草莓精油",
        name_en="Strawberry Essence",
        cas=None,
        category="Fruit",
        composition={
            "3913-81-3": 15.0,   # trans-2-Hexenal
            "105-54-4": 12.0,    # Ethyl butyrate
            "123-66-0": 10.0,    # Ethyl hexanoate
            "106-32-1": 8.0,     # Ethyl octanoate
            "123-92-2": 6.0,     # Isoamyl acetate
        },
        markers=["3913-81-3", "105-54-4"],
        sensory_notes=["草莓", "果味", "甜的", "酯香"]
    ),

    # Spice
    "ginger_oil": NaturalFingerprint(
        name_cn="姜油",
        name_en="Ginger Oil",
        cas="8007-08-7",
        category="Spice",
        composition={
            "23513-14-6": 25.0,  # Zingiberene (marker!)
            "495-60-3": 10.0,    # alpha-Curcumene
            "13474-59-4": 8.0,   # ar-Curcumene
            "470-82-6": 5.0,     # 1,8-Cineole
            "79-77-6": 3.0,      # beta-Ionone
        },
        markers=["23513-14-6"],  # Zingiberene
        sensory_notes=["姜", "辛辣", "温暖", "木质"]
    ),
}


class NaturalFingerprintDB:
    """
    Database manager for natural extract fingerprints.
    Supports both built-in data and user-provided fingerprints.
    """

    def __init__(self):
        self.fingerprints: Dict[str, NaturalFingerprint] = NATURAL_FINGERPRINTS.copy()
        self._marker_index: Dict[str, List[str]] = {}  # marker_cas -> [natural_ids]
        self._build_marker_index()

    def _build_marker_index(self):
        """Build reverse index from marker CAS to natural IDs."""
        self._marker_index = {}
        for nat_id, fp in self.fingerprints.items():
            for marker_cas in fp.markers:
                if marker_cas not in self._marker_index:
                    self._marker_index[marker_cas] = []
                self._marker_index[marker_cas].append(nat_id)

    def add_fingerprint(self, fp_id: str, fingerprint: NaturalFingerprint):
        """Add a user-defined fingerprint."""
        self.fingerprints[fp_id] = fingerprint
        self._build_marker_index()

    def get_by_id(self, fp_id: str) -> Optional[NaturalFingerprint]:
        """Get fingerprint by ID."""
        return self.fingerprints.get(fp_id)

    def get_by_marker(self, marker_cas: str) -> List[NaturalFingerprint]:
        """Get all naturals that have this marker."""
        nat_ids = self._marker_index.get(marker_cas, [])
        return [self.fingerprints[nid] for nid in nat_ids]

    def find_candidates(self, detected_cas_list: List[str]) -> Dict[str, float]:
        """
        Find natural extract candidates based on detected CAS numbers.
        Returns a dict of natural_id -> match_score (0-100).
        """
        scores = {}
        detected_set = set(detected_cas_list)

        for nat_id, fp in self.fingerprints.items():
            # Score based on how many composition components are detected
            composition_cas = set(fp.composition.keys())
            matched = composition_cas & detected_set
            if matched:
                # Weight by the percentage of matched components
                matched_pct = sum(fp.composition[cas] for cas in matched)
                total_pct = sum(fp.composition.values())
                score = (matched_pct / total_pct) * 100 if total_pct > 0 else 0

                # Bonus for marker matches
                marker_matches = set(fp.markers) & detected_set
                if marker_matches:
                    score = min(100, score + 20 * len(marker_matches))

                if score > 10:  # Only report meaningful matches
                    scores[nat_id] = round(score, 1)

        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    def get_all_naturals(self) -> List[NaturalFingerprint]:
        """Get all fingerprints."""
        return list(self.fingerprints.values())

    def save_to_json(self, filepath: str | Path):
        """Export fingerprints to JSON for persistence."""
        data = {}
        for nat_id, fp in self.fingerprints.items():
            data[nat_id] = {
                "name_cn": fp.name_cn,
                "name_en": fp.name_en,
                "cas": fp.cas,
                "category": fp.category,
                "composition": fp.composition,
                "markers": fp.markers,
                "sensory_notes": fp.sensory_notes
            }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_from_json(self, filepath: str | Path):
        """Load additional fingerprints from JSON."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for nat_id, fp_data in data.items():
            self.fingerprints[nat_id] = NaturalFingerprint(**fp_data)
        self._build_marker_index()


if __name__ == '__main__':
    # Test the fingerprint database
    db = NaturalFingerprintDB()
    print(f"Loaded {len(db.fingerprints)} natural fingerprints")
    print("\nCategories:", set(fp.category for fp in db.get_all_naturals()))

    # Test candidate finding
    test_cas = ["5989-27-5", "5392-40-5", "123-35-3"]  # Limonene, Citral, Myrcene
    candidates = db.find_candidates(test_cas)
    print(f"\nCandidates for {test_cas}:")
    for nat_id, score in candidates.items():
        fp = db.get_by_id(nat_id)
        print(f"  {fp.name_cn} ({fp.name_en}): {score}%")
