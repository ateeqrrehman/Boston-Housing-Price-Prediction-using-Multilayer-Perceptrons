# Boston-Housing-Price-Prediction-using-Multilayer-Perceptrons
Predicting Boston housing prices using Multilayer Perceptrons (MLPs) with scikit-learn and PyTorch. Regression with depth and width tuning.

**********************************************************************************************************************************************8
🧠 Boston Housing Price Prediction using Multilayer Perceptrons
This project explores the use of Multilayer Perceptrons (MLPs) to solve a regression problem using the classic Boston housing dataset. The dataset contains 13 input features describing various aspects of residential homes in Boston, and one output variable — MEDV (median value of owner-occupied homes in $1000s).

The goal is to build and evaluate a variety of feedforward neural networks, progressively increasing their complexity, and to understand how architectural choices (such as depth and width) affect model performance.

📊 Dataset Overview
The dataset includes 13 input variables:

CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT

The target variable:

MEDV — Median value of homes in $1000s

✅ Project Objectives
This project consists of several structured steps to build, tune, and evaluate MLP-based regression models.

Step 1 – Data Loading and Setup
Import necessary Python libraries (NumPy, PyTorch, scikit-learn, etc.).

Load the Boston housing dataset (housing.csv) into memory.

Set a random seed to ensure reproducibility across runs.

Step 2 – Data Preparation
Split the dataset into input features (X) and target output (y).

Convert data to appropriate numerical formats using NumPy.

Step 3 – Baseline Model with scikit-learn
Use scikit-learn’s Pipeline to build an MLPRegressor with two hidden layers.

Include StandardScaler in the pipeline to normalize features.

Use ReLU activation and Adam optimizer.

Step 4 – Standardization
Apply standardization within the cross-validation pipeline to avoid data leakage.

Confirm that the scaler is applied inside each training fold.

Step 5 – Model Evaluation via K-Fold Cross-Validation
Perform k-fold cross-validation (default k=10).

Report the Mean Squared Error (MSE) and its standard deviation.

Step 6 – PyTorch Model with a Single Hidden Layer
Build a PyTorch model with a single fully connected hidden layer and ReLU.

Sweep across a range of hidden unit counts (e.g., 4 to 256).

Observe how hidden layer width affects MSE and model performance.

Step 7 – Arbitrary-Depth PyTorch Network
Build a generalized PyTorch model supporting multiple hidden layers.

Use 32 neurons per hidden layer and vary depth from 1 to 5 layers.

Evaluate whether increased depth improves convergence or performance.

🎯 Performance Target
A reasonable model should achieve around 20 MSE (in $1000s²).
Note: scikit-learn's cross_val_score() returns negative MSE for optimization purposes; the absolute value should be used for interpretation.

⚙️ Environment Details
This project supports popular deep learning frameworks including PyTorch, TensorFlow, and Keras.
All code is implemented using PyTorch and scikit-learn.
To reproduce results, version information is printed at runtime. Sample versions used:

Python 3.10+

PyTorch 2.x

scikit-learn 1.4+

boston-mlp-regression/
├── boston_mlp.py          # Complete project code
├── housing.csv            # Boston dataset 
├── README.md              # Project overview (this file)
└── requirements.txt       # Library versions
