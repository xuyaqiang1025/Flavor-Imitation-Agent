# Flavor Imitation Agent - Comprehensive Strategy Overview

This document serves as the "Master Blueprint" for the project, confirming the end-to-end logic, technical architecture, and problem-solving strategies agreed upon.

## 1. Project Objective
Build an AI Agent capable of **reverse-engineering electronic cigarette flavors** with a focus on:
*   **High Fidelity**: Matching not just the chemical graph (GC-MS) but the sensory experience (Smell & Vape).
*   **Deconvolution**: Separating Natural Extracts (complex mixtures) from Synthetic Monomers.
*   **Vape Suitability**: Ensuring the formula performs well under atomization (no burnt notes, proper release).

---

## 2. Core Workflow & Technical Logic

### Step 1: Input & Smart Pre-processing
**Goal**: Convert raw, noisy instrument data into a clean "Chemical Concept".
*   **Inputs**:
    *   `Project Name` (e.g., "Blueberry Ice").
    *   `Sensory Description` (e.g., "Jammy, Sweet, High Cooling").
    *   `Raw GC-MS Data` (CSV/Excel Peak Table).
*   **The "Smart Cleaner"**:
    *   **Standardization**: Maps all peak names to **CAS Numbers** to ensure consistency.
    *   **Solvent Removal**: Auto-subtracts carrier solvents (PG, VG, Ethanol).
    *   **Reaction Product Filter (The "Virtual Reactor")**:
        *   *Logic*: The Agent knows that `Aldehydes + PG = Acetals`.
        *   *Action*: If it sees *Benzaldehyde Propylene Glycol Acetal*, it flags it as an **Artifact**, not an ingredient, and adds its area back to the parent Aldehyde if necessary.

### Step 2: The "Brain" - Natural vs. Synthetic Deconvolution
**Goal**: Solve the "Equation": `Target = (x% Natural A) + (y% Natural B) + (z% Synthetics)`.
*   **The "Dual-Track" Inference Strategy**:
    *   **Track A (Direct Match)**: If you provide a GC-MS of your own *Lemon Oil*, the Agent performs exact subtraction.
    *   **Track B (Probabilistic Inference - The Default)**:
        *   The Agent analyzes the "Feature Peaks" (e.g., Limonene, gamma-Terpinene, Citral).
        *   It queries its **Knowledge Graph**: *"What natural material typically contains these 3 molecules in this ratio?"*
        *   It calculates a probability: *"85% likelihood it's Lemon Oil, 15% Lime Oil."*
        *   It selects the best fit from your **Inventory**.
*   **Iterative Subtraction**:
    1.  Identify Natural #1 (e.g., Lemon Oil).
    2.  Calculate its contribution to all peaks.
    3.  **Subtract** those peaks from the Target Spectrum.
    4.  Repeat for Natural #2, #3...
    5.  Result: A "Residual Spectrum" representing the added **Synthetic Monomers**.

### Step 3: Formulation & Constraint Checking
**Goal**: Assembling the final recipe.
*   **Assembly**: Combine the identified Naturals + Residual Synthetics.
*   **Validation (Smell vs. Vape)**:
    *   **Physical Property Check**: The Agent checks `Boiling Point` and `LogP`.
    *   *Warning Logic*: "This molecule (e.g., a heavy lipid) has a BP of 400°C. It will not vaporize well and may cause carbon deposition (gunk). Suggest removing or replacing."

### Step 4: The Feedback Loop (Self-Correction)
**Goal**: Minimize the need for human trial-and-error.
*   **Internal Loop (The "AI Critic")**:
    1.  Agent drafts `Formula V1`.
    2.  **Simulator** predicts the GC-MS of `Formula V1` (Result A).
    3.  **Comparator** compares `Result A` vs. `Original Target`.
    4.  If the Error > Threshold, the Agent tweaks the formula *before* showing it to you.
*   **External Loop (User Feedback)**:
    *   You try the sample and say: *"Too simple, lacks a jammy note."*
    *   Agent translates "Jammy" -> `Ethyl 2-methylbutyrate` + `FEMA 200-300` descriptors -> Adjusts formula.

---

## 3. Deployment Roadmap
1.  **Phase 1: Knowledge Base Setup**
    *   Import your Inventory & FEMA data.
    *   Build the "Reaction Rules" for the cleaner.
2.  **Phase 2: The Engine**
    *   Implement the Subtraction Algorithm.
    *   Connect the LLM Inference for Naturals.
3.  **Phase 3: UI & Interaction**
    *   Simple interface to Upload CSV -> Get Recipe.

## 4. Execution Plan (Next Steps)
*   [ ] **Create `data/inventory/`**: We need to ingest your list.
*   [ ] **Build `cleaner.py`**: Start writing the code to parse your GC-MS files.
