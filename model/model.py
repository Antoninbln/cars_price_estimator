import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LinearRegression, Ridge
from sklearn.dummy import DummyRegressor

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def mean_absolute_percentage_error(Y_true, Y_pred): 
    y_true, y_pred = np.array(Y_true), np.array(Y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

class Model:
  def __init__(self, x, y, test_ratio_size=0.33):
    self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=test_ratio_size, random_state=0)
    self.y_pred = None

    # Modèles
    self.trained_model = None
    self.model = DummyRegressor(strategy="mean")
    ## Test more algo

  def fit(self):
    print("Training model...")
    self.trained_model = self.model.fit(self.x_train, self.y_train)
    print("Training done !")

  def predict(self, x_pred=None):
    """
    Predict target.

    Params : <type> - Description
    x_pred <DataFrame> - DataFrame to predict on. DEFAULT : None.
    NOTE: If exists it allows us to predict on real data, if not it allows to predict on test values.

    Returns : <type> - Description
    <DataFrame|Serie> - Target predicted
    """
    print("Predicting target...")
    if x_pred:
      self.y_pred = self.model.predict(self.x_pred)
    else:
      self.y_pred = self.model.predict(self.x_test)
    print("Prediction done !")

    return self.y_pred
  
  def get_result(self):
    # Mean squared error
    # MAPE
    print(f"MAPE -> {mean_absolute_percentage_error(self.y_test, self.y_pred)}")
    print(f"MAE -> {mean_absolute_error(self.y_test, self.y_pred)}")
