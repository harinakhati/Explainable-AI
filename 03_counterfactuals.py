# %% Imports

import importlib
import utils
importlib.reload(utils)

from utils import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score

import dice_ml


# %% Load and preprocess data

data_loader = DataLoader()

data_loader.load_dataset()
data_loader.preprocess_data()


# Split the data for evaluation
X_train, X_test, y_train, y_test = data_loader.get_data_split()


# Oversample the train data
X_train, y_train = data_loader.oversample(
    X_train,
    y_train
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

print("\nX_train data types:")
print(X_train.dtypes)

print("\nX_test data types:")
print(X_test.dtypes)


# %% Fit blackbox model

rf = RandomForestClassifier(
    random_state=2021
)

rf.fit(
    X_train,
    y_train
)

y_pred = rf.predict(X_test)

print(
    f"F1 Score {f1_score(y_test, y_pred, average='macro')}"
)

print(
    f"Accuracy {accuracy_score(y_test, y_pred)}"
)


# %% Create diverse counterfactual explanations

# Dataset
data_dice = dice_ml.Data(
    dataframe=data_loader.data,

    # All features are already numerically encoded.
    # Treat them as continuous to prevent DiCE
    # from converting True/False categorical values.
    continuous_features=list(
        data_loader.data.columns[:-1]
    ),

    outcome_name="stroke"
)


# Model
rf_dice = dice_ml.Model(
    model=rf,
    backend="sklearn"
)


explainer = dice_ml.Dice(
    data_dice,
    rf_dice,
    method="random"
)


# %% Create explanation

# Generate CF based on the blackbox model
input_datapoint = X_test.iloc[0:1]

print("\nInput datapoint:")
print(input_datapoint)

print("\nOriginal prediction:")
print(rf.predict(input_datapoint))


cf = explainer.generate_counterfactuals(
    input_datapoint,
    total_CFs=3,
    desired_class="opposite"
)


# Visualize it
cf.visualize_as_dataframe(
    show_only_changes=True
)


# %% Create feasible (conditional) Counterfactuals

features_to_vary = 'all' #[
     # 'avg_glucose_level',
     # 'bmi'
# ]

permitted_range = {
    'avg_glucose_level': [50, 250],
    'bmi': [18, 35]
}


# Now generating explanations
cf = explainer.generate_counterfactuals(
    input_datapoint,
    total_CFs=3,
    desired_class="opposite",
    permitted_range=permitted_range,
    features_to_vary=features_to_vary
)


# Visualize it
cf.visualize_as_dataframe(
    show_only_changes=True
)

# %%
