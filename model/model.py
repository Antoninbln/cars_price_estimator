import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.dummy import DummyRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import PowerTransformer

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def mean_absolute_percentage_error(Y_true, Y_pred): 
    y_true, y_pred = np.array(Y_true), np.array(Y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

class Model:
  def __init__(self, x, y, quali_features, test_ratio_size=0.33):
    self.x = self.categorize_cols(x, quali_features)

    # Split
    self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, y, test_size=test_ratio_size, random_state=0)
    
    print(f"Train shape : {self.x_train.shape}")
    self.y_pred = None
    self.y_pred_transformed = None

    # Modèles
    self.trained_model = None
    self.trained_model_transformed = None

    self.model = Ridge()
    self.model_transformed = TransformedTargetRegressor(regressor=Ridge(), transformer=PowerTransformer(method='box-cox')) # Box-cox because of the positive values

  def categorize_cols(self, input_df, quali_features):
    df = input_df

    for col in quali_features:
      if (col in df.columns.values):
        df[col] = df[col].astype('category')
        df[col] = df[col].cat.reorder_categories(df[col].unique(), ordered=True)
        df[col] = df[col].cat.codes

    return df

  def fit(self):
    print(f"Training on : {self.x_train.columns.values}")
    self.trained_model = self.model.fit(self.x_train, self.y_train)
    self.trained_model_transformed = self.model_transformed.fit(self.x_train, self.y_train)

  def predict(self, x_pred=None):
    """
    Predict target.

    Params : <type> - Description
    x_pred <DataFrame> - DataFrame to predict on. DEFAULT : None.
    NOTE: If exists it allows us to predict on real data, if not it allows to predict on test values.

    Returns : <type> - Description
    <DataFrame|Serie> - Target predicted
    """
    if x_pred:
      self.y_pred = self.model.predict(x_pred)
      self.y_pred_transformed = self.model_transformed.predict(x_pred)
    else:
      self.y_pred = self.model.predict(self.x_test)
      self.y_pred_transformed = self.model_transformed.predict(self.x_test)

    return self.y_pred, self.y_pred_transformed
  
  def get_result(self, should_print=True):
    # Mean squared error
    # MAPE
    not_tranformed = (mean_absolute_percentage_error(self.y_test, self.y_pred), r2_score(self.y_test, self.y_pred))
    tranformed = (mean_absolute_percentage_error(self.y_test, self.y_pred_transformed), r2_score(self.y_test, self.y_pred_transformed))

    print(f"\nResults not transformed data:")
    print(f"\tMAPE -> {not_tranformed[0]}")
    print(f"\tR2 -> {not_tranformed[1]}")
    print(f"Results transformed data:")
    print(f"\tMAPE -> {tranformed[0]}")
    print(f"\tR2 -> {tranformed[1]}")
    print("\n")

    return (not_tranformed, tranformed)
