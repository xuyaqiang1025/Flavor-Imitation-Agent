"""
Flavor Imitation Agent - LLM Client
=====================================
Handles interaction with LLM providers (e.g., OpenAI) for intelligent reasoning.
Replaces the rule-based translator with a semantic flavorist agent.
"""
import os
import json
from typing import List, Dict, Any
from dataclasses import dataclass

# Try importing openai, handle if not installed
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

@dataclass
class LLMAdjustment:
    target: str
    action: str
    amount: float
    reason: str

class FlavorLLMAgent:
    """
    LLM-powered flavorist agent.
    """
    def __init__(self, api_key: str = None, base_url: str = None, model: str = "gpt-4o"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        self.model = model
        self.client = None
        
        if self.api_key and OpenAI:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def is_configured(self) -> bool:
        return self.client is not None

    def suggest_adjustments(
        self, 
        user_feedback: str, 
        current_formula: List[Dict[str, Any]],
        inventory_summary: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Ask LLM for formula adjustments based on feedback.
        """
        if not self.client:
            return [{
                "target": "System", 
                "action": "Error", 
                "amount": 0, 
                "reason": "LLM API Key not configured. Please enter your API key in the sidebar."
            }]

        system_prompt = """
You are an expert Senior Flavorist specializing in e-cigarette e-liquids. 
Your task is to translate the user's sensory feedback into specific chemical formula adjustments.
You have a deep understanding of aroma chemicals, their thresholds, and interactions.

Current Formula context is provided.
The User Feedback describes the sensory gap (e.g., "Too sweet", "Lacks body", "Harsh throat hit").

Output ONLY a JSON array of adjustment objects with this schema:
[
  {
    "target": "Ingredient Name",  # Must match exactly an ingredient in the formula, or a generic class if adding new
    "action": "increase" | "decrease" | "remove" | "add",
    "amount": 0.5,  # The percentage CHANGE (absolute). E.g. 0.5 means +0.5% or -0.5% depending on action.
    "reason": "Scientific explanation why this change fixes the sensory issue"
  }
]
Do not include markdown formatting like ```json ... ```, just the raw JSON.
"""

        user_message = f"""
Current Formula:
{json.dumps(current_formula, ensure_ascii=False, indent=2)}

User Feedback: "{user_feedback}"

Inventory/Constraint Context: {inventory_summary}

Please analyze the feedback and suggest precise chemical adjustments.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3,
            )
            
            content = response.choices[0].message.content.strip()
            # Clean markdown if present
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "")
            
            adjustments = json.loads(content)
            return adjustments

        except Exception as e:
            return [{
                "target": "System",
                "action": "Error",
                "amount": 0,
                "reason": f"LLM Call Failed: {str(e)}"
            }]

if __name__ == "__main__":
    # Test
    agent = FlavorLLMAgent(api_key="sk-test") # This will fail connection but test init
    print("Agent initialized:", agent.is_configured())
