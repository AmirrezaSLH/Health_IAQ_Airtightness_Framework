## Thesis Link:
https://uwspace.uwaterloo.ca/items/21949c36-ad5c-4913-ad1b-5cae1466e5cf

# Health & Air Quality ROI Modeling Framework

This repository provides a modular Python-based analytical framework to assess the **return on investment (ROI)** of building interventions aimed at improving **indoor air quality (IAQ)** and **health outcomes**, both in historical and future scenarios. The project is structured into three core components:

- 📊 **Return_Investment**: ROI and health analysis based on indoor PM₂.₅ reduction.
- 🏠 **Indoor_Air**: Indoor exposure and IAQ modeling based on building attributes.
- 🔮 **Future_Scenario**: Projections and ROI analysis under future climate and policy scenarios.

---

## 📁 Folder Structure Overview

### 1. `Return_Investment/`

**Main Files**:
- `ROI_main.py`: Entry script for running deterministic or Monte Carlo ROI simulations.
- `ROI_Analysis.py`: Contains health/monetary modeling logic (e.g., VSL, mortality estimation).
- `plot_functions.py`: Shared plotting utilities (contour plots, histograms, box plots).

**Key Features**:
- Computes national and grid-level benefits and costs.
- Health model based on avoided PM₂.₅-related mortality and Value of Statistical Life (VSL).
- Monte Carlo sampling capabilities for uncertainty propagation.
- Generates spatial maps and summary boxplots.

**Strength**: Designed for static, point-in-time analysis with historical datasets.

---

### 2. `Indoor_Air/`

**Main Files**:
- `Indoor_main.py`: Orchestrates loading datasets, performing building-level simulations.
- `Indoor_Source.py`: Contains core logic for estimating indoor/outdoor PM₂.₅ contributions.

**Key Features**:
- Calculates PM₂.₅ infiltration using building air tightness (ACH₅₀), floor area, occupancy, and penetration/deposition constants.
- Supports:
  - Deterministic IAQ modeling
  - Indoor and outdoor source contribution decomposition
  - Exposure delta computation across segments

**Data Inputs**:
- Geospatial overlays of building stock, climate zones, and ACH₅₀ distributions.
- JSON-based dictionaries for segment-wise ACH₅₀, occupancy, and floor area.

**Strength**: Useful for IAQ exposure assessment under various retrofit configurations.

---

### 3. `Future_Scenario/`

**Main Files**:
- `future_main.py`: Data initialization and function definitions for PM₂.₅, building stock, population, etc.
- `Future_Analysis.py`: Contains scenario-specific logic and analysis (e.g., code compliance impacts).
- `plot_functions.py`: Used here for customized grouped boxplots and contour maps.

**Key Features**:
- Integrates future PM₂.₅ projections (e.g., REF, policy scenarios).
- Evaluates policy/code compliance interventions for Single-Family buildings by vintage (V1–V3).
- Generates ACH₅₀-adjusted Finf values per IECC climate zone.
- Supports spatial and national aggregation of impacts and cost-benefit.

**Strength**: Robust framework for forward-looking, scenario-based policy analysis.

---

## 🔄 Folder Comparison

| Feature                     | `Return_Investment` | `Indoor_Air`        | `Future_Scenario`       |
|----------------------------|---------------------|----------------------|--------------------------|
| Main focus                 | ROI, health outcome | IAQ and infiltration | Future scenarios, compliance |
| PM₂.₅ data source          | Historical average  | Historical & sampled | 2050–2100 scenario PM₂.₅ |
| Code compliance handling   | Optional            | Manual               | Climate-zone based logic |
| Monte Carlo support        | ✅                  | ✅                   | ✅                        |
| Plotting integration       | ✅                  | Partial              | ✅                        |
| Requires geospatial data   | ✅                  | ✅                   | ✅                        |

---

## 🚀 Getting Started

> **Note**: Ensure the required `Data/` directory exists with necessary `.json`, `.gpkg`, and `.csv` files.

### Requirements

Install dependencies (example using `pip`):
```bash
pip install pandas geopandas matplotlib shapely scipy
