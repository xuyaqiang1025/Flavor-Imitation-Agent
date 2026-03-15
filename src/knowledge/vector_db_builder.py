"""
Flavor Imitation Agent - Vector DB Builder
===========================================
Builds a local ChromaDB vector store containing natural extract fingerprints,
using either local JSON definitions or public literature sources.
"""
import os
import json
from pathlib import Path
from typing import List, Dict, Any

try:
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError:
    chromadb = None

# Built-in fallback data (if no external CSV is provided)
BUILTIN_NATURALS = [
    {
        "id": "lemon_oil",
        "name_cn": "柠檬精油 (Lemon Oil)",
        "description": "A citrus essential oil predominantly containing Limonene, beta-Pinene, and gamma-Terpinene, with trace citral.",
        "major_components": "Limonene (60-70%), beta-Pinene (10-15%), gamma-Terpinene (8-10%)",
        "markers": "Citral, Neryl acetate"
    },
    {
        "id": "grapefruit_oil",
        "name_cn": "葡萄柚精油 (Grapefruit Oil)",
        "description": "Citrus oil characterized by high Limonene and the presence of Nootkatone as a key marker.",
        "major_components": "Limonene (>90%), Myrcene (1-3%)",
        "markers": "Nootkatone"
    },
    {
        "id": "tobacco_absolute",
        "name_cn": "烟草浸膏 (Tobacco Absolute)",
        "description": "Rich, complex tobacco extract containing various nitrogenous compounds and degradation products of carotenoids.",
        "major_components": "Solanone, Megastigmatrienone, Damascenone",
        "markers": "Nicotine (trace), Quinoline, Trimethylpyrazine"
    },
    {
        "id": "peppermint_oil",
        "name_cn": "薄荷精油 (Peppermint Oil)",
        "description": "Cooling essential oil rich in menthol and menthone.",
        "major_components": "Menthol (30-50%), Menthone (15-30%), Menthyl acetate (2-10%)",
        "markers": "Menthol, Menthofuran"
    },
    {
        "id": "vanilla_extract",
        "name_cn": "香草提取物 (Vanilla Extract)",
        "description": "Sweet, creamy extract primarily defined by Vanillin.",
        "major_components": "Vanillin (>80% of volatiles)",
        "markers": "Vanillin, p-Hydroxybenzaldehyde"
    }
]


class VectorDBBuilder:
    def __init__(self, db_path: str = "./data/chroma_db"):
        self.db_path = db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        if chromadb is None:
            print("Warning: chromadb is not installed. Please `pip install chromadb`")
            self.client = None
        else:
            self.client = chromadb.PersistentClient(path=self.db_path)
            # Use default model (all-MiniLM-L6-v2) for sentence embeddings
            self.ef = embedding_functions.DefaultEmbeddingFunction()
            self.collection_name = "natural_extracts"
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name, 
                embedding_function=self.ef
            )

    def is_available(self) -> bool:
        return self.client is not None

    def build_from_builtin(self):
        """Populate the vector DB with built-in natural extract profiles."""
        if not self.is_available():
            return

        ids = []
        documents = []
        metadatas = []

        for item in BUILTIN_NATURALS:
            ids.append(item["id"])
            # Format document for semantic search
            doc = f"Extract Name: {item['name_cn']}. Description: {item['description']}. Major Components: {item['major_components']}. Marker Molecules: {item['markers']}."
            documents.append(doc)
            
            metadatas.append({
                "name": item["name_cn"],
                "major_components": item["major_components"],
                "markers": item["markers"]
            })

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        print(f"Successfully upserted {len(ids)} natural extracts into ChromaDB at {self.db_path}.")

    def search(self, query: str, n_results: int = 3) -> List[Dict]:
        """Search the vector DB for natural extracts matching a query."""
        if not self.is_available():
            return []
            
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        matches = []
        if results and results["ids"] and len(results["ids"]) > 0:
            for i in range(len(results["ids"][0])):
                matches.append({
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i] if "distances" in results else 0.0
                })
        return matches

if __name__ == "__main__":
    builder = VectorDBBuilder()
    builder.build_from_builtin()
    
    # Test search
    print("\nSearch Test: 'Limonene, Citral, Myrcene'")
    res = builder.search("Limonene, Citral, Myrcene")
    for r in res:
        print(f"Match: {r['metadata']['name']} (dist: {r['distance']:.3f})")
