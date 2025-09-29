# burned_embedder

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Aligning EO FM Embeddings for Burned Area detection

## Project Description

*burned_embedder* is a machine learning project that leverages Earth Observation (EO) foundation models to detect and map deforested areas from satellite imagery. The project uses the Copernicus Foundation Model to generate embeddings from Sentinel-1 radar data, which are then fine-tuned for the specific task of deforestation area detection.
Traditional deforestation detection methods often rely on optical imagery, which can be hindered by cloud cover and smoke. This project takes advantage of radar imagery (Sentinel-1) that can penetrate clouds, combined with state-of-the-art foundation model embeddings, to provide more reliable and accurate deforestation area mapping.

## Data Description

The project utilizes various data sources, including satellite imagery and ground truth data, to train models for burned area detection. In the following table, we summarize the key datasets used in this project:

| Dataset Name       | Description                                      | Source               |
|--------------------|--------------------------------------------------|----------------------|
| Radar Satellite Imagery  | Sentinel-1 SAR embeddings (Copernicus-FM)  | [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/dataset/sentinel-1-rtc)          |
| Optical Satellite Imagery  | Sentinel-2 High-resolution satellite images of the region (for visualization only)  | [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a)          |
| RADD               | Sentinel-1 derived "RAdar for Deforestation Alerts" | [Global Forest Watch](https://data.globalforestwatch.org/datasets/3c27123823a5461599fac9cb06862007_0/explore)        |
| Dynamic World V1   | near real-time LULC Dataset, 10m resolution from World Resources Institute and Google  | [Google Earth Engine](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1#description)                |


## Key Features

- Foundation Model Alignment: Fine-tunes Copernicus-FM embeddings specifically for deforestation area detection
- Multi-Source Data Integration: Combines Sentinel-1 radar, Sentinel-2 optical imagery, and reference datasets
- Cloud-Resilient Detection: Utilizes radar data that works in all weather conditions
- Scalable Pipeline: Built with modern data science best practices using Cookiecutter template

## Use Cases

- Deforestation monitoring
- Post-wildfire damage assessment
- Real-time fire monitoring and alert systems
- Historical burn scar mapping
- Environmental impact studies
- Forest management and conservation

## Workflow

┌─────────────────────────────────┐
│      Data Acquisition           │
│  • Sentinel-1 (Radar)           │
│  • Sentinel-2 (Optical)         │
│  • RADD Alerts                  │
│  • Dynamic World LULC           │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│     Data Preprocessing          │
│  • Image alignment              │
│  • Cloud masking                │
│  • Temporal compositing         │
│  • Label preparation            │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│     Feature Extraction          │
│  • Copernicus-FM embeddings     │
│  • Spatial features             │
│  • Temporal features            │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│      Model Training             │
│  • Fine-tune embeddings         │
│  • Train classifier             │
│  • Validate results             │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│   Inference & Evaluation        │
│  • Generate predictions         │
│  • Calculate metrics            │
│  • Create visualizations        │
└─────────────────────────────────┘

## Prerequisites

- Python 3.8+ (we used and recommend 3.12)
- CUDA-compatible GPU (recommended for training)
- Access to Microsoft Planetary Computer and Google Earth Engine

## Setup

### Clone the repository
`git clone https://github.com/luismaecker/burned_embedder`
`cd burned_embedder`

### Create virtual environment
`python -m venv venv`
`source venv/bin/activate`  # On Windows: `venv\Scripts\activate`

### Install dependencies
`pip install -r requirements.txt`

### Install the package in development mode
`pip install -e .`

## Project Organization

```
├── LICENSE            <- Open-source license
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         burned_embedder and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── burned_embedder   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes burned_embedder a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

## Acknowledgments

- Microsoft Planetary Computer for providing accessible satellite imagery
- Google Earth Engine for Dynamic World dataset
- Global Forest Watch for RADD alerts
- ESA, IBM and NASA for the Copernicus Foundation Model
- Cookiecutter Data Science for the project template
