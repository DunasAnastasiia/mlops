# Weather Prediction ML Project

Machine learning project for predicting rainfall in Australia using MLflow for experiment tracking.

## Project Structure

```
mlops_lab_1/
├── data/
│   └── raw/
│       └── weatherAUS.csv
├── notebooks/
│   └── 01_eda_weatherAUS.ipynb
├── src/
│   └── train.py
├── venv/
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
```

### 2. Activate Virtual Environment

Windows:
```bash
venv\Scripts\activate
```

Linux/Mac:
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Run Exploratory Data Analysis

```bash
jupyter notebook notebooks/01_eda_weatherAUS.ipynb
```

### Train Model

Basic usage:
```bash
python src/train.py
```

With custom parameters:
```bash
python src/train.py --model-type random_forest --n-estimators 100 --max-depth 15
python src/train.py --model-type logistic_regression --C 1.0
```

Available arguments:
- `--data-path`: Path to dataset (default: data/raw/weatherAUS.csv)
- `--model-type`: Model type (random_forest, logistic_regression)
- `--test-size`: Test set size (default: 0.2)
- `--n-estimators`: Number of trees for RandomForest
- `--max-depth`: Maximum depth for RandomForest
- `--min-samples-split`: Min samples split for RandomForest
- `--C`: Regularization parameter for LogisticRegression

### View MLflow Experiments

Launch MLflow UI:
```bash
mlflow ui
```

Open browser at: http://127.0.0.1:5000

## Results

| Experiment | Model | Parameters | Accuracy | ROC-AUC |
|------------|-------|------------|----------|---------|
| 1 | RandomForest | n=50, depth=10 | 0.8498 | 0.8727 |
| 2 | RandomForest | n=100, depth=15 | 0.8560 | 0.8845 |
| 3 | RandomForest | n=200, depth=20 | 0.8581 | 0.8875 |
| 4 | LogisticRegression | C=0.1 | 0.8449 | 0.8663 |
| 5 | LogisticRegression | C=1.0 | 0.8448 | 0.8663 |

Best model: RandomForest with 200 estimators and max_depth=20

## Dataset

Weather Australia dataset containing:
- 145,460 observations
- 23 features (weather measurements)
- Target: RainTomorrow (Yes/No)

Features include temperature, humidity, pressure, wind direction, rainfall, and more.

## Technologies

- Python 3.14
- scikit-learn
- MLflow
- pandas
- numpy
- matplotlib
- seaborn
- Jupyter
