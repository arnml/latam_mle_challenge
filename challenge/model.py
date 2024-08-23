from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DelayModel:

    def __init__(
        self
    ):
        checkpoint = joblib.load("./challenge/delay.joblib")
        # checkpoint = joblib.load("delay.joblib")
        self.__model = checkpoint["model"]
        self.__sklearn_version = checkpoint["sklearn_version"]
        self.__selected_features = checkpoint["selected_features"]
        self.__top_10_features = checkpoint["top_10_features"]
        self.threshold_in_minutes = 15

    # Getter and Setter for model
    @property
    def _model(self):
        return self.__model
    
    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, model):
        self.__model = model

    # Getter and Setter for sklearn_version
    @property
    def sklearn_version(self):
        return self.__sklearn_version

    @sklearn_version.setter
    def sklearn_version(self, sklearn_version):
        self.__sklearn_version = sklearn_version

    # Getter and Setter for selected_features
    @property
    def selected_features(self):
        return self.__selected_features

    @selected_features.setter
    def selected_features(self, selected_features):
        self.__selected_features = selected_features

    # Getter and Setter for top_10_features
    @property
    def top_10_features(self):
        return self.__top_10_features

    @top_10_features.setter
    def top_10_features(self, top_10_features):
        self.__top_10_features = top_10_features
    
    def __get_min_diff(self, data: pd.Series) -> float:
        fecha_o = datetime.strptime(data["Fecha-O"], "%Y-%m-%d %H:%M:%S")
        fecha_i = datetime.strptime(data["Fecha-I"], "%Y-%m-%d %H:%M:%S")
        min_diff = ((fecha_o - fecha_i).total_seconds()) / 60
        return min_diff

    def get_target(self, data: pd.DataFrame, target_column: str) -> pd.DataFrame:
        data["min_diff"] = data.apply(self.__get_min_diff, axis=1)
        data[target_column] = np.where(
            data["min_diff"] > self.threshold_in_minutes, 1, 0
        )
        return data[[target_column]]

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        filtered_data = data[self.selected_features]
        features = pd.concat([
            pd.get_dummies(filtered_data['OPERA'], prefix = 'OPERA'),
            pd.get_dummies(filtered_data['TIPOVUELO'], prefix = 'TIPOVUELO'), 
            pd.get_dummies(filtered_data['MES'], prefix = 'MES')], 
            axis = 1
        )
        features = features.reindex(columns=self.top_10_features, fill_value=0)


        if not target_column:
            return features

        return features, self.get_target(data=data, target_column=target_column)

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        x_train, _, y_train, _ = train_test_split(
            features, target, test_size=0.33, random_state=42
        )
        # features, target = self.preprocess(data = features, target_column=target)
        self.model.fit(x_train, y_train)
        return

    def predict(
        self,
        features: pd.DataFrame
    ) -> list[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        return self.model.predict(features).tolist()