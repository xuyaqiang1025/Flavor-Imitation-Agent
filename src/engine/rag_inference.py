"""
Flavor Imitation Agent - RAG Deconvolution Inference
====================================================
Uses a Retrieval-Augmented Generation approach to predict natural extracts
from GC-MS data via an LLM.
"""
import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from .deconvoluter import DeconvolutionResult

class RAGDeconvoluter:
    def __init__(self, vector_db_builder, api_key: Optional[str] = None, base_url: Optional[str] = None, model: str = "gpt-4o"):
        self.vector_db = vector_db_builder
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        self.model = model
        self.client = None
        
        if self.api_key and OpenAI:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def is_configured(self) -> bool:
        return self.client is not None and self.vector_db.is_available()

    def analyze(self, gcms_df: pd.DataFrame, max_retrieval: int = 4) -> DeconvolutionResult:
        """
        Using LLM to intelligently deconvolute the GC-MS data.
        """
        result = DeconvolutionResult()
        
        if not self.is_configured():
            result.warnings.append("API Key or Vector DB not configured. RAG Deconvolution unavailable.")
            return result

        if 'compound_name_cn' not in gcms_df.columns or len(gcms_df) == 0:
            result.warnings.append("No valid compound names in GC-MS data.")
            return result

        # 1. Prepare GC-MS Summary Context
        # Sort by concentration and take top 20 or anything above 1% to form a query profile
        total_conc = gcms_df['concentration_mg_kg'].sum()
        gcms_df['pct'] = gcms_df['concentration_mg_kg'] / total_conc * 100
        
        major_peaks = gcms_df[gcms_df['pct'] >= 0.5].sort_values('pct', ascending=False)
        peak_list = []
        for _, row in major_peaks.iterrows():
            name = row.get('compound_name_cn', str(row.get('cas', 'Unknown')))
            peak_list.append(f"{name} ({row['pct']:.1f}%)")
            
        peak_summary = ", ".join(peak_list)
        result.reasoning.append(f"Top detected peaks: {peak_summary}")

        # 2. Retrieve from Vector DB
        search_query = f"Match these molecules: {peak_summary}"
        retrieved_docs = self.vector_db.search(search_query, n_results=max_retrieval)
        
        if not retrieved_docs:
            result.warnings.append("Failed to retrieve any natural extract references from Vector DB.")
            context_str = "No database references found."
        else:
            context_lines = []
            for d in retrieved_docs:
                context_lines.append(f"- {d['document']}")
            context_str = "\n".join(context_lines)
            result.reasoning.append(f"Retrieved {len(retrieved_docs)} potential natural candidates from DB.")

        # 3. Ask LLM to reason the formulation
        system_prompt = """
You are an expert Flavor Chemist. Your goal is to deconvolute a GC-MS analysis of an e-cigarette flavor.
The "Deconvolution" process involves separating "Natural Extracts" (complex mixtures of many molecules) from "Synthetic Monomers" (single molecules added intentionally).

I will provide you with:
1. The GC-MS profile (Major peaks and their relative %).
2. A Reference Context (Database matches for natural extracts that MIGHT be present).

Your Task:
1. Determine IF any of the reference natural extracts are present in the GC-MS profile. Base this on the presence of marker molecules and expected composition ratios.
2. Estimate the roughly % of the final formula that is composed of each identified natural extract.
3. Deduce which of the REMAINING peaks must be "Synthetic Monomers" (added directly, not coming from the naturals).
4. Output your conclusion strictly as a JSON object.

Output JSON Schema:
{
  "reasoning": "Brief explanation of your deduction logic",
  "identified_naturals": {
    "Name of Natural (e.g., Lemon Oil)": 5.0,  // Estimated % in formula
  },
  "synthetic_monomers": [
    "Ethyl Maltol",
    "Vanillin"
    // List names of molecules you believe were added as pure synthetics, NOT from the naturals
  ]
}
"""

        user_message = f"""
GC-MS Profile (Top Peaks):
{peak_summary}

Reference Context (Vector DB Matches):
{context_str}

Please deconvolute this profile.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.2, 
            )
            
            content = response.choices[0].message.content.strip()
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "")
            
            data = json.loads(content)
            
            result.reasoning.append("LLM Reasoning: " + data.get("reasoning", "No info provided."))
            
            if "identified_naturals" in data:
                result.naturals = data["identified_naturals"]
                
             # Prepare synthetics dictionary based on the original dataframe
            if "synthetic_monomers" in data:
                synthetic_names = data["synthetic_monomers"]
                # We need to map names back to CAS and Concentrations
                for cas, row in gcms_df.set_index('cas').iterrows():
                    name_cn = row.get('compound_name_cn', '')
                    name_en = row.get('compound_name_en', '')
                    
                    is_synthetic = False
                    for syn_name in synthetic_names:
                        if syn_name.lower() in str(name_cn).lower() or syn_name.lower() in str(name_en).lower():
                            is_synthetic = True
                            break
                    
                    if is_synthetic:
                         result.synthetics[cas] = row['concentration_mg_kg']

            # Very rough confidence calculation based on naturals discovered + synthetics identified
            result.confidence = 80.0 if result.naturals else 50.0

            return result

        except Exception as e:
            result.warnings.append(f"LLM Reasoning Failed: {str(e)}")
            return result
