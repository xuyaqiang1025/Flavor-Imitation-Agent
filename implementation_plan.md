# Flavor Imitation Agent (仿香Agent) Implementation Plan

## Goal Description
Build an AI agent that assists flavor engineers in the e-cigarette industry to reverse-engineer and imitate target flavors. The system takes product names, sensory notes, and GC-MS data as input, and outputs a workable initial formula with adjustment suggestions. Key focus is solving the "Natural vs. Synthetic" deconvolution problem and ensuring consistency between "Smell" and "Vape" (Atomization) experience.

## User Review Required
> [!IMPORTANT]
> **Data Dependency - Natural Extract Fingerprints**: The "Subtraction/Deconvolution" logic requires knowing the chemical composition of the *User's* Natural Extracts.
> *   *Question*: Does your "Inventory Checklist" include the major peaks/composition for the naturals? (e.g., "Your Lemon Oil = 65% Limonene, 3% Citral...").
> *   *Mitigation*: If not, we will need to build a "Calibration" step where the agent learns these fingerprints from supplier COAs or public databases (Leffingwell, etc.), though public data may differ from your specific batch.

> [!WARNING]
> **Impurity Filtering**: Determining what is a "Reaction Product" vs. an "Intentional Ingredient" is complex. We will start with a Rule-Based Filter (Blocklist) that requires your domain expertise to populate (e.g., "Ignore peaks matching column bleed").

## Proposed Changes

## Proposed Changes (Revised)

### 1. Data Ingestion & Smart Cleaning `(src/ingestion)`
#### [NEW] `reaction_simulator.py` (Handling Q3 Impurities)
*   **Problem**: User lacks a list of reaction byproducts (e.g., Acetals formed by Aldehydes + PG/VG).
*   **Solution - "The Virtual Reactor"**:
    *   We will simulate standard organic chemistry rules: `Aldehyde + PG -> Acetal + H2O`.
    *   **Action**: Generate a dynamic "Blocklist" of likely acetals based on the *detected* aldehydes in the sample (e.g., if Vanillin is found, automatically flag "Vanillin Propylene Glycol Acetal" as a likely byproduct, not an ingredient).

### 2. Natural Extract Deconvolution - "Dual-Track" Strategy `(src/engine)`
#### [NEW] `natural_inference.py` (Handling Q1 Data Gap)
*   **Track A (High Precision)**: Uses "User Reference Data" (when available) for exact subtraction.
*   **Track B (LLM/Probabilistic - Immediate)**:
    *   **Logic**: Use an LLM-powered **Knowledge Graph**.
    *   *Input*: List of high-confidence peaks from Target.
    *   *Process*: Query LLM (with RAG on public databases like Leffingwell): "Which essential oils contain Nootkatone, Limonene, and Myrcene in ratio X:Y:Z?"
    *   *Output*: Probabilistic Ranking (e.g., "80% Grapefruit, 20% Lime").
    *   *Validation*: Check if User's Inventory contains "Grapefruit Oil".

### 3. Knowledge Base & Self-Correction `(src/knowledge)`
#### [NEW] `ai_critic.py` (Handling Q2 Labeling Fatigue)
*   **Function**: Acts as the "Internal Supervisor" to reduce human labeling.
*   **Workflow**:
    1.  Agent generates Candidate Formula.
    2.  **Critic** simulates the expected GC-MS of this candidate.
    3.  **Comparator** calculates "Reconstruction Error" vs. Target.
    4.  **Loop**: If error > Threshold, Agent tweaks formula *automatically* (e.g., "Add more Orange Oil to match Limonene peak").
    5.  **User Access**: Only disturb User when Self-Correction stalls (Error stops improving).

### 4. Logic & consistency Module `(src/logic)`
*   **Constraint Checking**:
    *   **Vape Compatibility**: Check *Solubility* (Polarity match with PG/VG) and *Thermal Stability* (avoid sugars/lipids that burn).

## Verification Plan

### Automated Tests
*   **Synthetic Mixture Test**: Create a "Virtual Target" composed of known inputs (e.g., 50% Lemon Oil + 50% Vanillin). Feed the *calculated* spectrum of this mix into the Agent. Verify if it returns the original 50/50 recipe.
*   **Sanity Check**: Verify that "Total Percentage" of output formula sums to roughly 100% (or identifies the missing solvent portion).

### Manual Verification
*   **Case Study**: User inputs a historical GC-MS file from a known successful project.
*   **Evaluation**: Compare Agent's output formula vs. the Engineer's actual final formula.
    *   *Success Metric*: Did the Agent identify the correct "Core Note" (e.g., Identified it was a Mango flavor)? Did it capture the top 3 high-impact ingredients?
