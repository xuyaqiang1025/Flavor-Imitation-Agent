"""
Flavor Imitation Agent - Vector DB Builder
===========================================
Builds a local ChromaDB vector store containing natural extract fingerprints.
Supports batch import from CSV, Excel, and Markdown files.
"""
import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError:
    chromadb = None

try:
    import pandas as pd
except ImportError:
    pd = None

# Built-in fallback data (baseline public-domain profiles)
BUILTIN_NATURALS = [
    {
        "id": "lemon_oil",
        "name_cn": "柠檬精油 (Lemon Oil)",
        "description": "A citrus essential oil predominantly containing Limonene, beta-Pinene, and gamma-Terpinene, with trace citral.",
        "major_components": "Limonene (60-70%), beta-Pinene (10-15%), gamma-Terpinene (8-10%)",
        "markers": "Citral, Neryl acetate",
        "composition": [
            {"cas": "5989-27-5", "name": "Limonene", "pct": 65.0},
            {"cas": "127-91-3", "name": "beta-Pinene", "pct": 12.0},
            {"cas": "99-85-4", "name": "gamma-Terpinene", "pct": 9.0},
            {"cas": "5392-40-5", "name": "Citral", "pct": 3.0},
        ]
    },
    {
        "id": "grapefruit_oil",
        "name_cn": "葡萄柚精油 (Grapefruit Oil)",
        "description": "Citrus oil characterized by very high Limonene with Nootkatone as a key unique marker.",
        "major_components": "Limonene (>90%), Myrcene (1-3%)",
        "markers": "Nootkatone",
        "composition": [
            {"cas": "5989-27-5", "name": "Limonene", "pct": 90.0},
            {"cas": "123-35-3", "name": "Myrcene", "pct": 2.0},
            {"cas": "4674-50-4", "name": "Nootkatone", "pct": 0.1},
        ]
    },
    {
        "id": "tobacco_absolute",
        "name_cn": "烟草浸膏 (Tobacco Absolute)",
        "description": "Rich, complex tobacco extract containing pyrazines, pyridines, quinoline, and carotenoid degradation products.",
        "major_components": "Solanone, Megastigmatrienone, Damascenone",
        "markers": "Quinoline, 2,3,5-Trimethylpyrazine",
        "composition": [
            {"cas": "91-22-5", "name": "Quinoline", "pct": 1.5},
            {"cas": "14667-55-1", "name": "2,3,5-Trimethylpyrazine", "pct": 5.0},
            {"cas": "22047-25-2", "name": "Acetylpyrazine", "pct": 2.0},
            {"cas": "23726-93-4", "name": "Damascenone", "pct": 0.3},
        ]
    },
    {
        "id": "peppermint_oil",
        "name_cn": "薄荷精油 (Peppermint Oil)",
        "description": "Cooling essential oil rich in menthol and menthone. Distinctive from spearmint by high menthol.",
        "major_components": "Menthol (30-50%), Menthone (15-30%), Menthyl acetate (2-10%)",
        "markers": "Menthol, Menthofuran",
        "composition": [
            {"cas": "89-78-1", "name": "Menthol", "pct": 42.0},
            {"cas": "89-80-5", "name": "Menthone", "pct": 22.0},
            {"cas": "16409-45-3", "name": "Menthyl acetate", "pct": 6.0},
        ]
    },
    {
        "id": "vanilla_extract",
        "name_cn": "香草提取物 (Vanilla Extract)",
        "description": "Sweet, creamy extract primarily defined by Vanillin alongside p-Hydroxybenzaldehyde.",
        "major_components": "Vanillin (>80% of volatiles)",
        "markers": "Vanillin, p-Hydroxybenzaldehyde",
        "composition": [
            {"cas": "121-33-5", "name": "Vanillin", "pct": 85.0},
            {"cas": "123-08-0", "name": "p-Hydroxybenzaldehyde", "pct": 3.0},
        ]
    },
]


class VectorDBBuilder:
    """
    Manages the local ChromaDB vector store for natural extract fingerprints.
    Supports upsert from built-in profiles, CSV, Excel (XLSX), and Markdown.
    """
    def __init__(self, db_path: Optional[str] = None):
        # Resolve db_path relative to the project root
        if db_path is None:
            # Default: <project_root>/data/chroma_db
            db_path = str(Path(__file__).parent.parent.parent / "data" / "chroma_db")
        self.db_path = db_path
        Path(self.db_path).mkdir(parents=True, exist_ok=True)

        if chromadb is None:
            print("Warning: chromadb is not installed. Please `pip install chromadb`")
            self.client = None
            self.collection = None
        else:
            self.client = chromadb.PersistentClient(path=self.db_path)
            self.ef = embedding_functions.DefaultEmbeddingFunction()
            self.collection_name = "natural_extracts"
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.ef
            )

    def is_available(self) -> bool:
        return self.client is not None

    def count(self) -> int:
        if not self.is_available():
            return 0
        return self.collection.count()

    # ------------------------------------------------------------------
    # Private: core upsert helper
    # ------------------------------------------------------------------
    def _upsert_items(self, items: List[Dict]) -> int:
        """
        Upsert a list of extract items into Chroma.

        Each item must have at minimum:
          id, name_cn, description, major_components, markers
        Optionally:
          composition (list of {cas, name, pct})
        """
        if not self.is_available():
            return 0

        ids, documents, metadatas = [], [], []
        for item in items:
            item_id = str(item.get("id", item.get("name_cn", ""))).lower().replace(" ", "_")
            name = item.get("name_cn", item.get("name", "Unknown"))
            description = item.get("description", "")
            major = item.get("major_components", "")
            markers = item.get("markers", "")

            # Compose the document text used for semantic search
            comp_items = item.get("composition", [])
            comp_str = ", ".join([f"{c['name']} ({c.get('pct', '?')}%)" for c in comp_items]) if comp_items else major

            doc = (
                f"Extract Name: {name}. "
                f"Description: {description}. "
                f"Major Components: {comp_str}. "
                f"Marker Molecules: {markers}."
            )

            # Metadata: store composition as JSON string for downstream quantitative math
            meta = {
                "name": name,
                "major_components": comp_str,
                "markers": markers,
                "composition_json": json.dumps(comp_items, ensure_ascii=False),
            }

            ids.append(item_id)
            documents.append(doc)
            metadatas.append(meta)

        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        return len(ids)

    # ------------------------------------------------------------------
    # Build from built-in baseline
    # ------------------------------------------------------------------
    def build_from_builtin(self) -> int:
        """Populate the vector DB with built-in public-domain profiles."""
        n = self._upsert_items(BUILTIN_NATURALS)
        print(f"[VectorDB] Upserted {n} built-in natural extract profiles.")
        return n

    # ------------------------------------------------------------------
    # Import from CSV
    # ------------------------------------------------------------------
    def import_from_csv(self, filepath: str) -> int:
        """
        Batch import natural extract profiles from a CSV file.
        Expected columns: id, name_cn, name_en, description, markers, composition_json
        `composition_json` should be a JSON array: [{cas, name, pct}, ...]
        """
        if pd is None:
            print("pandas is required. pip install pandas")
            return 0

        df = pd.read_csv(filepath)
        items = _dataframe_to_items(df)
        n = self._upsert_items(items)
        print(f"[VectorDB] Imported {n} records from CSV: {filepath}")
        return n

    # ------------------------------------------------------------------
    # Import from Excel
    # ------------------------------------------------------------------
    def import_from_excel(self, filepath: str, sheet_name: str = 0) -> int:
        """
        Batch import natural extract profiles from an Excel (.xlsx) file.
        Same column schema as CSV.
        """
        if pd is None:
            print("pandas is required. pip install pandas openpyxl")
            return 0

        df = pd.read_excel(filepath, sheet_name=sheet_name)
        items = _dataframe_to_items(df)
        n = self._upsert_items(items)
        print(f"[VectorDB] Imported {n} records from Excel: {filepath}")
        return n

    # ------------------------------------------------------------------
    # Import from Markdown
    # ------------------------------------------------------------------
    def import_from_markdown(self, filepath: str) -> int:
        """
        Batch import from a Markdown file using a specific H2 heading schema.

        Each extract should be formatted as:

        ## Extract Name (English / CN both OK)
        - **Description**: ...
        - **Markers**: ...
        - **Composition**:
          - Component Name (CAS:xxx-xx-x): XX%
          - ...
        """
        content = Path(filepath).read_text(encoding="utf-8")

        # Split by H2 headings
        sections = re.split(r'\n## ', content)
        items = []
        for section in sections:
            if not section.strip():
                continue
            lines = section.strip().splitlines()
            name = lines[0].strip().lstrip("# ")
            item_id = name.lower().replace(" ", "_").replace("(", "").replace(")", "")[:60]
            description = ""
            markers = ""
            comp_items = []

            in_composition = False
            for line in lines[1:]:
                line = line.strip()
                if line.startswith("- **Description**:"):
                    description = line.replace("- **Description**:", "").strip()
                    in_composition = False
                elif line.startswith("- **Markers**:"):
                    markers = line.replace("- **Markers**:", "").strip()
                    in_composition = False
                elif line.startswith("- **Composition**"):
                    in_composition = True
                elif in_composition and line.startswith("-"):
                    # e.g. "  - Limonene (CAS:5989-27-5): 65%"
                    comp_line = line.lstrip("- ").strip()
                    cas_match = re.search(r'CAS[:\s]*([\d-]+)', comp_line, re.IGNORECASE)
                    pct_match = re.search(r'([\d.]+)\s*%', comp_line)
                    comp_name = re.sub(r'\s*\(.*\)', '', comp_line).strip()
                    if cas_match and pct_match:
                        comp_items.append({
                            "cas": cas_match.group(1),
                            "name": comp_name,
                            "pct": float(pct_match.group(1)),
                        })

            items.append({
                "id": item_id,
                "name_cn": name,
                "description": description,
                "markers": markers,
                "composition": comp_items,
                "major_components": ", ".join(
                    [f"{c['name']} ({c['pct']}%)" for c in sorted(comp_items, key=lambda x: -x['pct'])[:5]]
                ),
            })

        n = self._upsert_items(items)
        print(f"[VectorDB] Imported {n} records from Markdown: {filepath}")
        return n

    # ------------------------------------------------------------------
    # Semantic search (used by RAG engine)
    # ------------------------------------------------------------------
    def search(self, query: str, n_results: int = 3) -> List[Dict]:
        """Search the vector DB for natural extracts semantically matching a query."""
        if not self.is_available() or self.count() == 0:
            return []

        actual_n = min(n_results, self.count())
        results = self.collection.query(
            query_texts=[query],
            n_results=actual_n,
        )

        matches = []
        if results and results["ids"] and len(results["ids"]) > 0:
            for i in range(len(results["ids"][0])):
                meta = results["metadatas"][0][i]
                # Re-inflate composition from JSON string
                comp_json = meta.get("composition_json", "[]")
                try:
                    composition = json.loads(comp_json)
                except Exception:
                    composition = []

                matches.append({
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": meta,
                    "composition": composition,  # List of {cas, name, pct}
                    "distance": results["distances"][0][i] if "distances" in results else 0.0,
                })
        return matches

    # ------------------------------------------------------------------
    # CAS matrix similarity search (Track A enhanced)
    # ------------------------------------------------------------------
    def find_candidates_by_cas_vector(self, cas_to_conc: Dict[str, float], top_k: int = 5) -> List[Dict]:
        """
        Variable-dimension cosine similarity matching between the GC-MS fingerprint
        and all stored extract profiles.

        Each extract is represented as a sparse vector over all CAS numbers it contains.
        The input is also treated as a sparse vector.
        This works for any number of components (not fixed 50 dimensions).
        """
        if not self.is_available() or self.count() == 0:
            return []

        # Retrieve all entries
        all_results = self.collection.get(include=["metadatas", "documents"])
        all_ids = all_results["ids"]
        all_metas = all_results["metadatas"]

        scored = []
        total_target = sum(cas_to_conc.values())
        if total_target == 0:
            return []

        # Normalise target vector
        target_vec = {cas: conc / total_target for cas, conc in cas_to_conc.items()}

        for idx, meta in enumerate(all_metas):
            comp_json = meta.get("composition_json", "[]")
            try:
                composition = json.loads(comp_json)
            except Exception:
                continue

            if not composition:
                continue

            # Build reference vector (normalised % -> fraction)
            ref_vec = {c["cas"]: c["pct"] / 100.0 for c in composition if c.get("cas") and c.get("pct")}
            if not ref_vec:
                continue

            # Cosine similarity over the shared CAS space
            common_cas = set(target_vec.keys()) & set(ref_vec.keys())
            if not common_cas:
                continue

            dot = sum(target_vec[c] * ref_vec[c] for c in common_cas)
            mag_target = sum(v ** 2 for v in target_vec.values()) ** 0.5
            mag_ref = sum(v ** 2 for v in ref_vec.values()) ** 0.5

            if mag_target == 0 or mag_ref == 0:
                continue

            score = dot / (mag_target * mag_ref)

            scored.append({
                "id": all_ids[idx],
                "name": meta.get("name", "Unknown"),
                "composition": composition,
                "cosine_similarity": round(score, 4),
                "markers": meta.get("markers", ""),
            })

        scored.sort(key=lambda x: -x["cosine_similarity"])
        return scored[:top_k]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _dataframe_to_items(df) -> List[Dict]:
    """Convert a DataFrame (from CSV or Excel) into a list of extract item dicts."""
    items = []
    for _, row in df.iterrows():
        comp_json_raw = str(row.get("composition_json", "[]"))
        try:
            composition = json.loads(comp_json_raw)
        except Exception:
            composition = []

        items.append({
            "id": str(row.get("id", row.get("name_cn", ""))).lower().replace(" ", "_"),
            "name_cn": str(row.get("name_cn", row.get("name_en", "Unknown"))),
            "description": str(row.get("description", "")),
            "markers": str(row.get("markers", "")),
            "composition": composition,
            "major_components": ", ".join(
                [f"{c['name']} ({c.get('pct', '?')}%)" for c in sorted(composition, key=lambda x: -float(x.get('pct', 0)))[:5]]
            ) if composition else str(row.get("major_components", "")),
        })
    return items


if __name__ == "__main__":
    builder = VectorDBBuilder()
    builder.build_from_builtin()

    print(f"DB now has {builder.count()} entries.")
    print("\nSemantic Search: 'Limonene, Citral, Myrcene'")
    res = builder.search("Limonene, Citral, Myrcene")
    for r in res:
        print(f"  {r['metadata']['name']} (dist={r['distance']:.3f})")

    print("\nCAS Vector Search: {Limonene: 65%, beta-Pinene: 12%}")
    cas_res = builder.find_candidates_by_cas_vector({"5989-27-5": 65.0, "127-91-3": 12.0})
    for r in cas_res:
        print(f"  {r['name']} (cosine={r['cosine_similarity']:.3f})")
