MLOps homework1
==============================

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

Initially use
```bash
pip install -e .
```

For train use
```bash
train
```
or 
```bash
train -cn train_config_1
train -cn train_config_2
```

To predict use 
```bash
predict
```

To make random detaset use
```bash
make_dataset
```

To run tests use
```bash
python test/test_all.py
```
Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── configs
    │   │
    │   ├── config.py         
    │   │
    │   ├── make_dataset.yaml
    │   │
    │   ├── predict_cofig.yaml
    │   │
    │   ├── train_cofig_1.yaml
    │   │
    │   └── train_cofig_2.yaml
    ├── data
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks         
    │    └── EDA.ipynb        
    │                         
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── report.html 
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
    │       └── make_report.py
    │
    ├── test
    │   │
    │   ├── test_all.py         
    │   │
    │   ├── test_make_dataset.py
    │   │
    │   ├── test_predict.py
    │   │
    │   └── test_train.py
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io
