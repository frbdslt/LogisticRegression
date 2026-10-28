# Heart Disease Prediction with Logistic Regression

A small educational project that implements binary logistic regression using NumPy and gradient descent. It loads the heart disease dataset, removes duplicate rows, splits the data into training, validation, and test sets, standardizes the features, trains the model, and reports test metrics.

## Project structure

```text
.
├── data/
│   └── heart.csv
├── src/
│   ├── logistic.py
│   └── preprocessing.py
├── main.py
└── requirements.txt
```

## Requirements

- Python 3
- Packages listed in `requirements.txt`

The listed package versions match the environment used to inspect this project.

## Setup and run

From the project directory, create and activate a virtual environment, install dependencies, and run the program:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py
```

On macOS or Linux, activate the environment with `source .venv/bin/activate` instead.

The program reads `data/heart.csv` using a path relative to the project directory. It prints the sample counts and test metrics, then displays training-loss and validation-accuracy plots.

## Data and features

The CSV contains 1,025 rows before duplicate removal. `LoadAndCleanData` drops exact duplicate rows, leaving 302 unique samples in the current dataset. The target column is `target`; the model uses these six input columns:

- `cp`
- `thalach`
- `ca`
- `oldpeak`
- `exang`
- `thal`

The features are standardized using the training-set mean and standard deviation. A bias column is added for the intercept.

## Training and evaluation

The data split is shuffled with a fixed seed (`42`) and uses 20% for testing and 15% for validation; the remaining samples are used for training. The logistic regression implementation uses binary cross-entropy loss and batch gradient descent. The reported evaluation includes accuracy, precision, recall, F1 score, and a confusion matrix.

The validation accuracy and training loss are plotted across optimization steps. The stopping condition is based on the change in training loss.

## Note

This is an educational machine-learning example, not a clinical diagnostic tool.
