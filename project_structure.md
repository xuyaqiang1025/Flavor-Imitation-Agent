# Flavor Imitation Agent - Project Structure & Communication

## 1. Directory Structure (Updated)

```
Flavor Imitation Agent/
├── docs/                           # Documentation
│   ├── comprehensive_strategy.md   # Master strategy document
│   └── task.md                     # Progress checklist
├── data/
│   ├── raw_gcms/                   # Input: GC-MS export files
│   │   └── Sample1.csv             # Example tobacco flavor
│   └── inventory/                  # Raw material databases
│       └── sample_inventory.csv    # Generated sample (15 materials)
├── src/                            # Source Code
│   ├── main.py                     # ★ ENTRY POINT - Run this!
│   ├── ingestion/                  # Data Reading
│   │   ├── parser.py               # GC-MS file reader
│   │   └── cleaner.py              # Noise filter + Virtual Reactor
│   ├── knowledge/                  # Databases
│   │   ├── inventory_manager.py    # Raw material lookup
│   │   └── natural_fingerprints.py # Essential oil compositions (9 built-in)
│   ├── engine/                     # Core Logic
│   │   └── deconvoluter.py         # Iterative Subtraction algorithm
│   └── logic/                      # Validation & Feedback
│       ├── sensory_validator.py    # Smell vs Vape checks
│       └── translator.py           # Feedback -> Chemistry
├── implementation_plan.md          # Technical roadmap
└── project_structure.md            # This file
```

## 2. How to Run

```bash
cd "Flavor Imitation Agent"
python src/main.py
```

This will:
1. Parse the sample GC-MS file
2. Clean noise and predict artifacts
3. Run deconvolution to find naturals vs synthetics
4. Generate a formula with percentages
5. Validate for vape compatibility

## 3. Module Communication (Data Flow)

```
User Input                    Knowledge Bases
    │                              │
    ▼                              ▼
┌─────────┐   ┌─────────┐   ┌──────────────┐
│ GC-MS   │──▶│ Parser  │──▶│ Cleaner      │
│ CSV     │   │         │   │ (blocklist)  │
└─────────┘   └─────────┘   └──────┬───────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
            ┌───────────┐  ┌────────────┐  ┌───────────┐
            │ Inventory │  │ Fingerprint│  │ Deconvolu-│
            │ Manager   │  │ Database   │  │ ter       │
            └─────┬─────┘  └──────┬─────┘  └─────┬─────┘
                  │               │              │
                  └───────────────┴──────────────┘
                                   │
                                   ▼
                          ┌───────────────┐
                          │ Formula       │
                          │ Generator     │
                          └───────┬───────┘
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
            ┌───────────┐  ┌────────────┐  ┌───────────┐
            │ Sensory   │  │ Translator │  │ Output    │
            │ Validator │  │ (Feedback) │  │ Formula   │
            └───────────┘  └────────────┘  └───────────┘
```

## 4. Key Files Explained

| File | Purpose |
|------|---------|
| `main.py` | Entry point, orchestrates the full pipeline |
| `parser.py` | Reads GC-MS CSV, handles Chinese headers |
| `cleaner.py` | Removes solvents, nicotine, predicts acetals |
| `natural_fingerprints.py` | 9 essential oils with composition data |
| `deconvoluter.py` | Core algorithm: separates naturals from synthetics |
| `sensory_validator.py` | Checks if ingredients work for vaping |
| `translator.py` | Converts "too sweet" -> "reduce maltol" |

## 5. Next Steps for User

1. **Add your real inventory** - Replace `sample_inventory.csv` with your actual raw material list
2. **Add natural fingerprints** - When you get GC-MS of your oils, add them to `natural_fingerprints.py`
3. **Test with your samples** - Put your GC-MS exports in `data/raw_gcms/`
