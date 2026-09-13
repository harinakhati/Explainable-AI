# Explainable AI Experiments

This is an evolving practice project for exploring explainable artificial
intelligence (XAI) methods across tabular and image datasets. The examples
currently cover healthcare stroke prediction and brain MRI classification, and
the project will grow as new datasets, models, and explanation techniques are
added.

## Methods

### Current tabular example: stroke prediction

The tabular examples use the healthcare stroke dataset in
`data/healthcare-dataset-stroke-data.csv`. The shared `DataLoader` preprocesses
categorical features with one-hot encoding, fills missing BMI values, splits
the data, and balances the training set with `RandomOverSampler`.

- `data_exploration.py` explores the dataset and plots feature histograms.
- `interpretable_models.py` trains and explains Logistic Regression,
	Classification Tree, and Explainable Boosting Machine models with
	InterpretML.
- `01_lime.py` trains a Random Forest and creates local LIME explanations.
- `02_shap.py` trains a Random Forest and creates local and global SHAP
	explanations.
- `03_counterfactuals.py` generates diverse and constrained counterfactuals
	with DiCE.

### Brain MRI classification and LRP

`lrp_brain_mri.ipynb` trains a VGG16-based PyTorch classifier on images stored
under `data/brain_mri/training/` and `data/brain_mri/testing/`. It visualizes
model predictions and applies a custom Layer-wise Relevance Propagation (LRP)
implementation to produce an MRI relevance heatmap.

The current LRP implementation is written directly with PyTorch operations;
it does not require a separate `lrp` package. `captum` is included in the
requirements for possible future attribution experiments.

### Linear regression for bike-sharing interpretability

`01-linear-regression-bike-sharing-interpretability.ipynb` implements the linear
regression example from the *Interpretable Machine Learning* book using the UCI
bike-sharing daily dataset at `data/bike+sharing+dataset/day.csv`. The notebook
preprocesses the data, adds a two-day lag feature, encodes seasonal and weather
categories with reference levels, and evaluates a scikit-learn linear regression
model with MAE, RMSE, and test $R^2$.

It also fits the training data with `statsmodels.OLS` to inspect coefficient
uncertainty and creates weight plots, feature-effect plots, and an individual
prediction decomposition. The notebook focuses on reproducing the book's data
processing and interpretation concepts; its coefficients may differ because it
uses a reproducible `train_test_split` with `random_state=42`.

### Logistic regression for penguin interpretability

`02-logistic-regression.ipynb` reproduces the logistic regression example from
the *Interpretable Machine Learning* book using the Palmer Penguins dataset at
`data/penguins.csv`. It filters to Chinstrap penguins, removes incomplete rows,
and predicts whether a penguin is female from bill measurements, flipper
length, and a categorical `chonkiness` feature derived from body mass.

The notebook uses `statsmodels.Logit` to examine coefficients, standard errors,
odds ratios, confidence intervals, and p-values. It also decomposes an
individual prediction into log-odds contributions and evaluates the fitted
model with classification metrics, ROC-AUC, a calibration curve, and the Brier
score. The reported ROC-AUC (about `0.974`) and Brier score (about `0.067`) are
training-set results, so they should not be treated as estimates of
generalization performance.

### Experiments based on *Interpretable Machine Learning*

This project follows Christoph Molnar's [*Interpretable Machine Learning*]
(https://christophm.github.io/interpretable-ml-book/) book as a practical study
guide. The bike-sharing notebook is the first implementation of its interpretable
linear-model concepts, including the bike-sharing and penguin logistic
regression notebooks. Future experiments will work through the book's broader
coverage of interpretable models, model-agnostic methods, feature effects,
surrogate models, and example datasets. This includes the [Logistic Regression]
(https://christophm.github.io/interpretable-ml-book/logistic.html) chapter, as
well as further chapters and techniques selected as the project develops.

## Project structure

The structure below describes the current examples. New datasets can be added
under `data/`, with dataset-specific preprocessing and experiments kept in
separate scripts or notebooks.

```text
.
├── Tutorial-practiced/
│   ├── 01_lime.py
│   ├── 02_shap.py
│   ├── 03_counterfactuals.py
│   ├── data_exploration.py
│   ├── interpretable_models.py
│   ├── utils.py
│   └── lrp_brain_mri.ipynb
├── 01-linear-regression-bike-sharing-interpretability.ipynb
├── 02-logistic-regression.ipynb
├── data/
│   ├── healthcare-dataset-stroke-data.csv
│   ├── penguins.csv
│   └── brain_mri/
│       ├── training/
│       └── testing/
├── requirements.txt
└── README.md
```

## Git and ignored files

The repository keeps source code, notebooks, documentation, and dependency
files under version control. The `.gitignore` excludes local or generated
files:

- `.vscode/` and `venv/` environment-specific configuration and environments.
- `.ipynb_checkpoints/` generated Jupyter notebook checkpoints.
- `__pycache__/` Python bytecode caches.
- `data/`, `*.csv`, and `*.jpg` datasets and image files, which may be large
	or subject to their own licensing and privacy restrictions.

The datasets described in this README are therefore expected to exist locally
but are not committed to Git. Check each dataset's source, license, and access
requirements before adding it to the project.

## Environment and setup

The project uses a Conda environment named `myenv` with Python 3.10.20 or
later within the Python 3.10 series. Create and activate it with:

```bash
conda create -n myenv python=3.10.20
conda activate myenv
```

Verify that the active terminal is using the expected environment:

```bash
python --version
which python
```

Install the project dependencies after activating `myenv`:

```bash
python -m pip install -r requirements.txt
```

The requirements pin PyTorch 2.2.2 and torchvision 0.17.2 and constrain
NumPy to version 1.x for compatibility with that PyTorch release.

For VS Code notebooks, select the `myenv` interpreter or the `myenv` Jupyter
kernel. The selected interpreter should report Python 3.10.20 when running
`python --version`. Leave the environment with:

```bash
conda deactivate
```

## Running the current tabular examples

Run scripts from the project root so the relative dataset path resolves:

```bash
python Tutorial-practiced/data_exploration.py
python Tutorial-practiced/interpretable_models.py
python Tutorial-practiced/01_lime.py
python Tutorial-practiced/02_shap.py
python Tutorial-practiced/03_counterfactuals.py
```

InterpretML and some explanation methods open visualizations through the
Python session or notebook environment. The current scripts use the relative
path `data/healthcare-dataset-stroke-data.csv`.

## Adding future experiments

When introducing another dataset:

1. Add the data under `data/` and document its source and expected format.
2. Add or adapt preprocessing so feature types, missing values, and the target
	variable are handled explicitly.
3. Create a clearly named script or notebook for the model and explanation
	method.
4. Add any new direct package dependencies to `requirements.txt`.
5. Update this README with the new dataset, command, and any special setup.

Keep reusable data preparation and evaluation code in shared modules when it
applies to more than one experiment.

## Ongoing work and future directions

This project is being developed as a hands-on collection of XAI experiments.
Planned areas of practice include:

- Applying the existing XAI methods to additional healthcare and general
	machine-learning datasets.
- Comparing model performance and explanation stability across datasets and
	model types.
- Adding more explanation methods, including permutation importance,
	partial dependence, and additional counterfactual approaches.
- Improving the custom PyTorch LRP implementation and comparing its heatmaps
	with established attribution tools such as Captum.
- Making preprocessing, evaluation, and visualization utilities more
	reusable across experiments.
- Recording experiment settings, metrics, and observations so results can be
	compared consistently.

## Running the notebooks

Open `01-linear-regression-bike-sharing-interpretability.ipynb` in VS Code or
Jupyter and select the `myenv` Python kernel. Run the cells from the repository
root so the relative path to `data/bike+sharing+dataset/day.csv` resolves.

Open `02-logistic-regression.ipynb` in VS Code or Jupyter and select the `myenv`
Python kernel. Run the cells from the repository root so the relative path to
`data/penguins.csv` resolves. The notebook drops rows with missing values and
evaluates the model on the same observations used for fitting; use a held-out
split or cross-validation for a generalization estimate.

Open `Tutorial-practiced/lrp_brain_mri.ipynb` in VS Code or Jupyter and select
the `myenv` Python kernel. The notebook was originally configured for Google
Colab and GPU execution. Before running it locally, update `TRAIN_ROOT` and
`TEST_ROOT` to the local `data/brain_mri/training` and
`data/brain_mri/testing` directories.

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

The pretrained VGG16 weights may need to be downloaded the first time the
model is created, so network access is required for that initial run.

## Notes

- Run the tabular scripts from the repository root.
- The current scripts expect the data files to remain in the paths shown above.
- Model scores and explanation outputs can vary when a model does not specify
	a random seed.
