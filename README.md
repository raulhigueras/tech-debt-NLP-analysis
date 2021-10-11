# tech-debt-NLP-analysis

Natural Language Processing analysis of developers' written messages in the Technical Debt Database to predict issues' difficulty.

## Table of Content

1. [Project Status](#project-status)
2. [Instructions of Use](#instructions-of-use)
3. [Project Organization](#project-organization)


## Project Status

- [x] Problem Understanding
- [x] Data Exploration 
- [x] Data Preprocessing
- [ ] Modeling
- [ ] Evaluation
- [ ] Deploy


## Instructions of use

### Setup
First of all, run the following command to install all the python libraries needed.
```{bash}
make requirements
```

### Data Exploration
The data exploration part is all contained in the notebook `001-exploration.ipynb`. To see the whole analysis, just run the notebook from the start of the document.

> ❗️ Warning
>
> Make sure to include both versions of the databases (`td_V1.bd` and `td_V2.bd`) in the folder `data/raw`.

### Data Preprocessing 
To generate the processed dataset, run the following command:
```{bash}
make data
```

The NLP preprocessing techniques are detailed and explained in the notebook `002-preprocessing.ipynb`. Those techniques are implemented just after the modelling step, and can be executed following the instructions at [Modeling](###modeling).

### 🚧 Modeling 🚧
To generate the features, run the followig command:
```{bash}
make features
```

### 🚧 Evaluation 🚧
WIP

### 🚧 Deploy 🚧
WIP

## Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
