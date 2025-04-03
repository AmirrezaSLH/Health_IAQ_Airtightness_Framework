## Thesis Link:
https://uwspace.uwaterloo.ca/items/21949c36-ad5c-4913-ad1b-5cae1466e5cf

# Health & Air Quality ROI Modeling Framework

This repository provides a modular Python-based analytical framework to assess the **return on investment (ROI)** of building interventions aimed at improving **indoor air quality (IAQ)** and **health outcomes**, both in historical and future scenarios. The main intervention studied is improving airtightness, however the model can be modified to cover a wide range of interventions. The project is structured into three core components:

- 📊 **Return_Investment**: ROI and health analysis based on indoor PM₂.₅ reduction of outdoor origin.
- 🏠 **Indoor_Air**: Indoor exposure and IAQ modeling, accounting for both indoor-generated and outdoor-origin particles. (including the effects of cooking.)
- 🔮 **Future_Scenario**: Projections and ROI analysis under future climate and policy scenarios.

---

## 📁 Folder Structure Overview

**Data Inputs**:
- Geospatial overlays of building stock, climate zones, and ACH₅₀ distributions.
- JSON-based dictionaries for segment-wise ACH₅₀, occupancy, and floor area.
- `plot_functions.py`: Shared plotting utilities (contour plots, histograms, box plots).

### 1. `Return_Investment/`

**Main Files**:
- `ROI_main.py`: Entry script for running deterministic or Monte Carlo ROI simulations. This file loads data, manages the data pipelines and do all the calculations.
- `ROI_Analysis.py`: Defines intervention scenarios, calls the main simulation file, saves outputs, call plot_functions to generate plots.


**Key Features**:
- Calculates PM₂.₅ infiltration using building air tightness (ACH₅₀), floor area, occupancy, and penetration/deposition constants.
- Generates ACH₅₀-adjusted Finf values per IECC climate zone.
- Health model based on avoided PM₂.₅-related mortality and Value of Statistical Life (VSL).
- Computes national and grid-level benefits and costs.
- Monte Carlo sampling capabilities for uncertainty propagation.
- Supports spatial and national aggregation of impacts and cost-benefit.
- Generates spatial maps and summary boxplots.

---

### 2. `Indoor_Air/`

**Main Files**:
- `Indoor_main.py`: In addition to ROI_main.py, it incorporates the impacts of range hood usage and cooking into the model. It can also be extended to simulate other indoor-generated particle sources.
- `Indoor_Source.py`: Defines intervention scenarios and cooking scenarios, calls the main simulation file, saves outputs, call plot_functions to generate plots.

**Key Features**:
- Supports:
  - Deterministic indoor generated source modeling
  - Indoor and outdoor source contribution decomposition
  - Exposure delta computation across segments

**Strength**: Useful for IAQ exposure assessment under various retrofit configurations.

---

### 3. `Future_Scenario/`

**Main Files**:
- `future_main.py`: Data initialization and function definitions for PM₂.₅, population, baseline mortality and VSL for future scenarios.
- `Future_Analysis.py`: Contains scenario-specific logic and analysis (e.g., code compliance impacts).
- `plot_functions.py`: Used here for customized grouped boxplots.

**Key Features**:
- Integrates future PM₂.₅ projections (e.g., REF, policy scenarios).
- Evaluates policy/code compliance interventions for Single-Family buildings.

**Strength**: Robust framework for forward-looking, scenario-based policy analysis.

---

## 🔄 Folder Comparison

| Feature                     | `Return_Investment` | `Indoor_Air`        | `Future_Scenario`       |
|----------------------------|---------------------|----------------------|--------------------------|
| Main focus                 | ROI of airtightness improvements | Impacts of Indoor Generated Sources | Future scenarios, compliance |
| PM₂.₅ data source          | Historical  | Historical and Literature | 2050–2100 scenario PM₂.₅ |
| Monte Carlo support        | ✅                  | Partial                   | ✅                        |
| Plotting integration       | ✅                  | ✅             | ✅                        |

---

## 🚀 Getting Started

> **Note**: Ensure the required `Data/` directory exists with necessary `.json`, `.gpkg`, and `.csv` files.

### Requirements

Install dependencies (example using `pip`):
```bash
pip install pandas geopandas matplotlib shapely scipy
