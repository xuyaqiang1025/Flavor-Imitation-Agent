"""
Flavor Imitation Agent - Hybrid Deconvolution Engine
=====================================================
Combines:
  Track A  : CAS-matrix cosine similarity (math, fast, no API)
  Track B  : RAG + LLM qualitative identification ("which naturals?")
  Track C  : Math-quantitative subtraction ("how much of each?")

Virtual Reactor Traceback:
  Detected acetal -> reverse-calculate parent aldehyde amount -> add back to profile
"""
import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from .deconvoluter import DeconvolutionResult

# ------------------------------------------------------------------ #
# Known acetal -> parent aldehyde mapping for traceback calculation   #
# Format: acetal_cas -> (aldehyde_cas, aldehyde_mw, acetal_mw)       #
# ------------------------------------------------------------------ #
ACETAL_TRACEBACK = {
    "68527-74-2": ("121-33-5", 152.15, 226.27),   # Vanillin PG Acetal -> Vanillin
    # Extend as new reaction products are confirmed
}


class HybridDeconvoluter:
    """
    Hybrid deconvolution engine.
    Step 1 - Traceback: acetal peaks are converted back to parent aldehyde amounts.
    Step 2 - CAS Matrix: cosine similarity to get initial candidates (no LLM).
    Step 3 - LLM Qualitative: confirm WHICH naturals are present via RAG + LLM.
    Step 4 - Math Quantitative: calculate HOW MUCH each natural contributes via
             marker back-calculation, then subtract from spectrum.
    Step 5 - Residuals are labeled as synthetic monomers.
    """

    def __init__(
        self,
        vector_db_builder,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "deepseek-chat",
    ):
        self.vector_db = vector_db_builder
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        self.model = model
        self.client = None

        if self.api_key and OpenAI:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def is_configured(self) -> bool:
        return self.client is not None and self.vector_db.is_available()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def analyze(self, gcms_df: pd.DataFrame, max_retrieval: int = 5) -> DeconvolutionResult:
        result = DeconvolutionResult()

        if not self.is_configured():
            result.warnings.append("API Key or Vector DB not configured. Hybrid Deconvolution unavailable.")
            return result

        if len(gcms_df) == 0:
            result.warnings.append("Empty GC-MS DataFrame.")
            return result

        # Work on a copy with a running concentration column
        df = gcms_df.copy().reset_index(drop=True)
        if "concentration_mg_kg" not in df.columns:
            result.warnings.append("No concentration_mg_kg column.")
            return result

        # ------------------------------------------------------------------
        # STEP 1: Virtual Reactor Traceback
        # ------------------------------------------------------------------
        df, traceback_notes = self._reactor_traceback(df)
        result.reasoning.extend(traceback_notes)

        total_conc = df["concentration_mg_kg"].sum()
        if total_conc == 0:
            result.warnings.append("Total concentration is 0 after traceback.")
            return result

        # Build working dict: cas -> remaining concentration
        cas_to_remaining = {}
        for _, row in df.iterrows():
            cas = str(row.get("cas", "")).strip()
            if cas and cas not in ("nan", "0-00-0", ""):
                cas_to_remaining[cas] = cas_to_remaining.get(cas, 0) + float(row["concentration_mg_kg"])

        # ------------------------------------------------------------------
        # STEP 2: CAS Matrix Similarity (no LLM)
        # ------------------------------------------------------------------
        vector_candidates = self.vector_db.find_candidates_by_cas_vector(cas_to_remaining, top_k=max_retrieval)
        if vector_candidates:
            top_names = ", ".join([f"{c['name']} ({c['cosine_similarity']:.2f})" for c in vector_candidates[:3]])
            result.reasoning.append(f"[CAS Matrix] Top candidates: {top_names}")

        # ------------------------------------------------------------------
        # STEP 3: RAG + LLM Qualitative Identification
        # ------------------------------------------------------------------
        confirmed_naturals = self._llm_qualify(df, vector_candidates, total_conc, result)

        # ------------------------------------------------------------------
        # STEP 4: Math Quantitative Subtraction
        # ------------------------------------------------------------------
        for nat_name, nat_composition in confirmed_naturals:
            amount_mg_kg, basis = self._calc_amount_math(nat_composition, cas_to_remaining)
            if amount_mg_kg <= 0:
                result.warnings.append(f"Could not calculate amount for {nat_name}, skipping.")
                continue

            pct = amount_mg_kg / total_conc * 100
            result.naturals[nat_name] = round(pct, 2)
            result.reasoning.append(
                f"[Math] {nat_name}: {pct:.2f}% "
                f"(≈{amount_mg_kg:.0f} mg/kg, basis: {basis})"
            )

            # Subtract natural's contribution from the remaining pool
            for comp in nat_composition:
                cas = comp.get("cas", "")
                pct_in_nat = float(comp.get("pct", 0)) / 100.0
                contrib = amount_mg_kg * pct_in_nat
                if cas in cas_to_remaining:
                    cas_to_remaining[cas] = max(0.0, cas_to_remaining[cas] - contrib)

        # ------------------------------------------------------------------
        # STEP 5: Residuals = Synthetic Monomers
        # ------------------------------------------------------------------
        threshold = total_conc * 0.005  # ignore peaks < 0.5% of total
        for cas, rem_conc in cas_to_remaining.items():
            if rem_conc > threshold:
                result.synthetics[cas] = rem_conc

        # Confidence
        explained_nat = sum(result.naturals.values())
        residual_pct = sum(result.synthetics.values()) / total_conc * 100 if total_conc > 0 else 0
        result.confidence = min(100.0, round(explained_nat + residual_pct, 1))

        if result.confidence < 50:
            result.warnings.append("低置信度: 仍有较大比例未能归因，可能存在未知天然物。")
        if not result.naturals:
            result.warnings.append("未识别到明显的天然提取物，配方可能为纯合成。")

        return result

    # ------------------------------------------------------------------
    # STEP 1 helper: Virtual Reactor Traceback
    # ------------------------------------------------------------------
    def _reactor_traceback(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Detect known acetal peaks and add back the parent aldehyde molar equivalent
        to the running concentration, then zero-out the acetal peak.

        Acetal -> Aldehyde mass conversion:
            aldehyde_mass = acetal_mass * (aldehyde_MW / acetal_MW)
        """
        notes = []
        if "cas" not in df.columns:
            return df, notes

        for acetal_cas, (ald_cas, ald_mw, acetal_mw) in ACETAL_TRACEBACK.items():
            acetal_rows = df[df["cas"] == acetal_cas]
            if acetal_rows.empty:
                continue

            total_acetal_conc = acetal_rows["concentration_mg_kg"].sum()
            recovered_ald_conc = total_acetal_conc * (ald_mw / acetal_mw)

            # Add recovered aldehyde concentration back
            ald_rows = df[df["cas"] == ald_cas]
            if not ald_rows.empty:
                df.loc[df["cas"] == ald_cas, "concentration_mg_kg"] += recovered_ald_conc
                notes.append(
                    f"[Traceback] 检测到缩醛 {acetal_cas} ({total_acetal_conc:.1f} mg/kg) "
                    f"→ 还原亲本醛 {ald_cas} +{recovered_ald_conc:.1f} mg/kg"
                )
            else:
                # Parent aldehyde not detected at all, insert a synthetic row
                new_row = acetal_rows.iloc[0].copy()
                new_row["cas"] = ald_cas
                new_row["concentration_mg_kg"] = recovered_ald_conc
                new_row["compound_name_cn"] = f"[Traceback还原] {ald_cas}"
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                notes.append(
                    f"[Traceback] 插入还原醛 {ald_cas}: {recovered_ald_conc:.1f} mg/kg"
                )

            # Zero-out the acetal peak (it was a reaction product, not an ingredient)
            df.loc[df["cas"] == acetal_cas, "concentration_mg_kg"] = 0.0

        return df, notes

    # ------------------------------------------------------------------
    # STEP 3 helper: LLM Qualitative
    # ------------------------------------------------------------------
    def _llm_qualify(
        self,
        df: pd.DataFrame,
        vector_candidates: List[Dict],
        total_conc: float,
        result: DeconvolutionResult,
    ) -> List[Tuple[str, List[Dict]]]:
        """
        Ask LLM: given the GC-MS profile and the database candidates,
        confirm WHICH naturals are actually present (yes/no per candidate).
        Returns: list of (name, composition) for confirmed naturals.
        """
        if not vector_candidates:
            result.warnings.append("No vector DB candidates found – skipping LLM step.")
            return []

        # Build peak list context
        df_sorted = df.copy()
        df_sorted["pct"] = df_sorted["concentration_mg_kg"] / total_conc * 100
        top_peaks = df_sorted[df_sorted["pct"] >= 0.3].sort_values("pct", ascending=False).head(30)
        peak_lines = []
        for _, r in top_peaks.iterrows():
            name = r.get("compound_name_cn") or r.get("compound_name_en") or str(r.get("cas", "?"))
            peak_lines.append(f"  {name} ({r['pct']:.2f}%)")
        peak_summary = "\n".join(peak_lines)

        # Build candidate context (from vector DB hits)
        cand_lines = []
        for c in vector_candidates:
            markers = c.get("markers", "")
            comp_str = ", ".join([f"{x['name']} ({x['pct']}%)" for x in c.get("composition", [])[:6]])
            cand_lines.append(
                f"  - {c['name']} (cosine={c['cosine_similarity']:.2f}): "
                f"Markers=[{markers}], Components=[{comp_str}]"
            )
        cand_summary = "\n".join(cand_lines)

        system_prompt = """You are a Senior Flavor Chemist specializing in GC-MS deconvolution.

Your ONLY task here is QUALITATIVE: determine which natural extracts from the candidate list are ACTUALLY PRESENT in the sample.

Rules:
1. A natural extract is "confirmed" ONLY if its specific MARKER molecule(s) appear in the GC-MS profile.
2. If the marker is absent but common components are present, classify as "possible" not "confirmed".
3. The percentage field is for your rough ESTIMATE – a separate math step will compute it precisely. So don't stress about exact numbers.

Output ONLY a JSON array:
[
  {
    "name": "Tobacco Absolute",
    "confirmed": true,
    "estimated_pct": 3.0,
    "key_evidence": "Quinoline and Trimethylpyrazine both detected"
  }
]
No markdown, just raw JSON."""

        user_message = f"""GC-MS Profile:
{peak_summary}

Database Candidates:
{cand_summary}

Which of these candidates are confirmed present?"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.1,
            )
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            qualifications = json.loads(content)

            confirmed = []
            for q in qualifications:
                if q.get("confirmed"):
                    # Find the matching candidate to get its composition
                    match_comp = []
                    for c in vector_candidates:
                        if c["name"].lower() in q["name"].lower() or q["name"].lower() in c["name"].lower():
                            match_comp = c.get("composition", [])
                            break
                    confirmed.append((q["name"], match_comp))
                    result.reasoning.append(
                        f"[LLM ✓] {q['name']}: {q.get('key_evidence', '')} "
                        f"(initial est. {q.get('estimated_pct', '?')}%)"
                    )
            return confirmed

        except Exception as e:
            result.warnings.append(f"LLM qualification failed: {str(e)}")
            return []

    # ------------------------------------------------------------------
    # STEP 4 helper: Math quantitative back-calculation
    # ------------------------------------------------------------------
    def _calc_amount_math(
        self,
        composition: List[Dict],
        cas_to_remaining: Dict[str, float],
    ) -> Tuple[float, str]:
        """
        Back-calculate the amount of a natural extract using marker/major component logic.

        For each component with a known CAS and percent-in-natural:
            natural_amount = detected_amount / (pct_in_natural / 100)

        Takes the median across all components to buffer outliers.
        Returns (amount_mg_kg, basis_description).
        """
        if not composition:
            return 0.0, "no composition data"

        estimates = []
        for comp in composition:
            cas = comp.get("cas", "")
            pct_in_nat = float(comp.get("pct", 0))
            if not cas or pct_in_nat <= 0:
                continue
            detected = cas_to_remaining.get(cas, 0.0)
            if detected <= 0:
                continue
            estimate = detected / (pct_in_nat / 100.0)
            estimates.append((estimate, comp.get("name", cas)))

        if not estimates:
            return 0.0, "no matching CAS in spectrum"

        # Use median to dampen outliers
        estimates.sort(key=lambda x: x[0])
        mid = len(estimates) // 2
        amount, basis_name = estimates[mid]
        return amount, f"component {basis_name}"
