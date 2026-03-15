"""
Flavor Imitation Agent - Virtual Reactor Agent
==============================================
Uses LLM to intelligent predict chemical reaction byproducts (e.g., Acetals, Esters)
that occur naturally during the aging (steeping) of e-liquids, so they can
be properly filtered out during deconvolution.
"""
import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

class VirtualReactorAgent:
    """
    LLM-powered virtual reactor to predict artifacts based on chemical rules.
    """
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: str = "gpt-4o"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        self.model = model
        self.client = None
        
        if self.api_key and OpenAI:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def is_configured(self) -> bool:
        return self.client is not None

    def predict_reactions(self, df: pd.DataFrame, solvents: List[str] = ["Propylene Glycol (PG)", "Vegetable Glycerin (VG)", "Ethanol"]) -> List[Dict[str, Any]]:
        """
        Ask LLM to predict reaction byproducts based on the detected molecules.
        
        Args:
            df: Cleaned GC-MS DataFrame containing 'cas' and 'compound_name_cn'.
            solvents: List of primary solvents in the matrix that might participate in reactions.
            
        Returns:
            List of dictionary describing the predicted byproducts.
        """
        if not self.is_configured():
            return [{"error": "LLM API not configured for Virtual Reactor."}]
            
        if 'compound_name_cn' not in df.columns or len(df) == 0:
            return []

        # Extract list of compounds for the LLM
        compounds = df['compound_name_cn'].dropna().unique().tolist()
        
        # If there are too many (e.g. > 100), maybe take the top 100 by concentration
        if len(compounds) > 100 and 'concentration_mg_kg' in df.columns:
            top_df = df.sort_values('concentration_mg_kg', ascending=False)
            compounds = top_df['compound_name_cn'].dropna().unique().tolist()[:100]

        system_prompt = """
You are an expert Organic Chemist specializing in flavor chemistry and e-liquid aging (steeping).
E-liquids contain flavor molecules and bulk solvents (e.g., Propylene Glycol, Vegetable Glycerin).
Over time, reactions can occur, forming byproducts. 
Common reactions:
1. Aldehyde + Propylene Glycol -> PG Acetal + H2O
2. Acid + Alcohol -> Ester + H2O 

Your task is to review a provided list of detected flavor molecules and predict IF any specific, well-known reaction products are highly likely to form with the given bulk solvents.

Output ONLY a JSON array of predicted byproduct objects.
Schema:
[
  {
    "parent_molecule": "Vanillin",
    "reaction_type": "Acetalization",
    "predicted_byproduct_name": "Vanillin propylene glycol acetal",
    "reason": "Vanillin is an aldehyde that readily forms an acetal with Propylene Glycol"
  }
]
Do not include any formatting like ```json, just output the raw JSON array.
If no obvious reactions are likely, output [].
"""

        user_message = f"""
Detected Flavor Molecules:
{json.dumps(compounds, ensure_ascii=False)}

Bulk Solvents in System:
{json.dumps(solvents, ensure_ascii=False)}

Please predict the likely reaction byproducts.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.1, # Low temperature for more deterministic chemistry facts
            )
            
            content = response.choices[0].message.content.strip()
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "")
            
            predictions = json.loads(content)
            # Ensure it is a list
            if isinstance(predictions, list):
                return predictions
            else:
                return []

        except Exception as e:
            return [{"error": f"LLM Call Failed: {str(e)}"}]

if __name__ == "__main__":
    # Test agent
    agent = VirtualReactorAgent("test_key")
    print("Is configured:", agent.is_configured())
