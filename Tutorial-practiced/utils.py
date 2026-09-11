import pandas as pd

# Makes sure we see all columns
pd.set_option('display.max_columns', None)

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler


class DataLoader():

    def __init__(self):
        self.data = None

    def load_dataset(
        self,
        path="data/healthcare-dataset-stroke-data.csv"
    ):
        self.data = pd.read_csv(path)

    def preprocess_data(self):

        # One-hot encode all categorical columns
        categorical_cols = [
            "gender",
            "ever_married",
            "work_type",
            "Residence_type",
            "smoking_status"
        ]

        encoded = pd.get_dummies(
            self.data[categorical_cols],
            prefix=categorical_cols,
            dtype=int
        )

        # Update data with new columns
        self.data = pd.concat(
            [encoded, self.data],
            axis=1
        )

        self.data.drop(
            categorical_cols,
            axis=1,
            inplace=True
        )

        # Impute missing values of BMI
        self.data["bmi"] = self.data["bmi"].fillna(0)

        # Drop id as it is not relevant
        self.data.drop(
            ["id"],
            axis=1,
            inplace=True
        )

        # Make sure all features are numeric
        feature_columns = self.data.columns[:-1]

        self.data[feature_columns] = (
            self.data[feature_columns].apply(pd.to_numeric)
        )

    def get_data_split(self):

        X = self.data.iloc[:, :-1]
        y = self.data.iloc[:, -1]

        return train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=2021,
            stratify=y
        )

    def oversample(self, X_train, y_train):

        oversample = RandomOverSampler(
            sampling_strategy='minority',
            random_state=2021
        )

        # Oversample pandas DataFrame directly
        X_over, y_over = oversample.fit_resample(
            X_train,
            y_train
        )

        # Make absolutely sure the model receives numeric data
        X_over = X_over.astype(float)

        return X_over, y_over