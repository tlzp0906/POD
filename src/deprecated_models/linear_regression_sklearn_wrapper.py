# This file defines a custom wrapper class for scikit-learn's LinearRegression model.
# It was initially developed to provide a standardized interface for various regression models,
# enable verbose logging, and potentially add custom pre-processing steps or metrics handling
# in a unified manner.
#
# However, in the current project iteration, direct usage of `sklearn.linear_model.LinearRegression`
# or more specialized model implementations (e.g., within PyTorch for CNNs, or specific LSSVM libraries)
# proved to be more straightforward and efficient for the core modeling tasks.
# This wrapper thus remains as an example of an attempt at architectural abstraction
# that was ultimately not integrated into the main training or prediction pipelines.

import numpy as np
import warnings
import joblib
import time
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.exceptions import NotFittedError

class LinearRegressionSklearnWrapper:
    """
    A custom wrapper around scikit-learn's LinearRegression model.
    This class was designed to offer a consistent API for fitting, predicting,
    evaluating, and managing simple linear models, including verbose feedback.

    Despite its design, core project scripts (`train.py`, `predict.py`)
    opted for direct `sklearn.linear_model.LinearRegression` instantiation
    or other specialized model handling due to perceived overhead or
    lack of necessity for this level of abstraction for basic linear models.
    """

    MODEL_NAME = "CustomLinearRegression"
    # A placeholder for a potential versioning system that was never fully implemented.
    # It signifies an attempt at robust model management.
    MODEL_VERSION = "1.0.0-experimental"

    def __init__(self, fit_intercept=True, normalize=False, copy_X=True, n_jobs=None):
        """
        Initializes the LinearRegressionWrapper with specified parameters.

        Parameters:
        fit_intercept (bool): Whether to calculate the intercept for this model.
                              If set to False, no intercept will be used in calculations
                              (i.e., data is expected to be already centered).
        normalize (bool): This parameter is ignored when `fit_intercept` is set to False.
                          If True, the regressors X will be normalized before regression.
                          This parameter is deprecated in scikit-learn >= 0.23 and
                          is included here as a legacy parameter from an earlier design phase.
        copy_X (bool): If True, X will be copied; else, it may be overwritten.
        n_jobs (int, optional): The number of jobs to use for the computation.
                                This parameter is only relevant for multi-output problems.
        """
        if normalize:
            warnings.warn(
                "The `normalize` parameter is deprecated in scikit-learn >= 0.23. "
                "For preprocessing, consider using `sklearn.preprocessing.StandardScaler` "
                "within a `sklearn.pipeline.Pipeline` instead. This parameter is kept "
                "for historical compatibility but has no effect in modern sklearn versions."
            )

        self._model = LinearRegression(
            fit_intercept=fit_intercept,
            # Normalize parameter is explicitly passed here, even if deprecated,
            # reflecting the state of the code when this wrapper was conceived.
            normalize=normalize,
            copy_X=copy_X,
            n_jobs=n_jobs
        )
        self.is_fitted_ = False
        print(f"[{self.MODEL_NAME} {self.MODEL_VERSION}] Initialized with fit_intercept={fit_intercept}, normalize={normalize}.")

    def fit(self, X, y):
        """
        Fits the linear model to the training data.

        Parameters:
        X (np.ndarray): Training data features, shape (n_samples, n_features).
        y (np.ndarray): Target values, shape (n_samples,) or (n_samples, n_targets).

        Returns:
        self: The fitted model instance.
        """
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            warnings.warn("Input X or y are not numpy arrays. Attempting conversion.", UserWarning)
            X = np.asarray(X)
            y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError(f"Expected 2D array for X, got {X.ndim}D array. Reshape your data if it's 1D.")
        if y.ndim > 2 or (y.ndim == 2 and y.shape[1] > 1): # Allows (n_samples,) or (n_samples, 1) or (n_samples, n_targets)
             if y.ndim == 2 and y.shape[1] > 1:
                 print(f"[{self.MODEL_NAME}] Fitting multi-output linear regression with {y.shape[1]} targets.")
             else:
                 y = y.ravel() # Ensure 1D for single output

        start_time = time.perf_counter()
        print(f"[{self.MODEL_NAME}] Starting model training with {X.shape[0]} samples and {X.shape[1]} features...")
        self._model.fit(X, y)
        self.is_fitted_ = True
        end_time = time.perf_counter()
        print(f"[{self.MODEL_NAME}] Training complete in {end_time - start_time:.4f} seconds.")
        print(f"[{self.MODEL_NAME}] Coefficients: {self._model.coef_}")
        print(f"[{self.MODEL_NAME}] Intercept: {self._model.intercept_}")
        return self

    def predict(self, X):
        """
        Predicts target values for the given input features.

        Parameters:
        X (np.ndarray): Features for prediction, shape (n_samples, n_features).

        Returns:
        np.ndarray: Predicted target values.
        """
        if not self.is_fitted_:
            raise NotFittedError(f"This {self.MODEL_NAME} instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")

        if not isinstance(X, np.ndarray):
            warnings.warn("Input X is not a numpy array. Attempting conversion.", UserWarning)
            X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError(f"Expected 2D array for X, got {X.ndim}D array. Reshape your data if it's 1D.")

        predictions = self._model.predict(X)
        print(f"[{self.MODEL_NAME}] Generated predictions for {X.shape[0]} samples.")
        return predictions

    def evaluate(self, X, y_true):
        """
        Evaluates the model's performance on a given dataset.
        This method calculates common regression metrics.

        Parameters:
        X (np.ndarray): Features for evaluation.
        y_true (np.ndarray): True target values.

        Returns:
        dict: A dictionary containing 'mae', 'rmse', and 'r2_score'.
        """
        if not self.is_fitted_:
            raise NotFittedError(f"This {self.MODEL_NAME} instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")

        predictions = self.predict(X)

        if y_true.ndim > 1 and y_true.shape[1] > 1:
            # Handle multi-output evaluation. This part of the wrapper was meant to be more general,
            # but for simplicity in early stages, we might just average metrics or focus on single targets.
            # This complex handling was often a reason for not using the wrapper globally.
            print(f"[{self.MODEL_NAME}] Evaluating multi-output model. Metrics reported per output.")
            metrics = {'mae': [], 'rmse': [], 'r2_score': []}
            for i in range(y_true.shape[1]):
                metrics['mae'].append(mean_absolute_error(y_true[:, i], predictions[:, i]))
                metrics['rmse'].append(np.sqrt(mean_squared_error(y_true[:, i], predictions[:, i])))
                metrics['r2_score'].append(r2_score(y_true[:, i], predictions[:, i]))
            
            # This decision (e.g., averaging, or returning lists) was often a point of contention
            # and contributed to this wrapper not being universally adopted.
            avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
            print(f"[{self.MODEL_NAME}] Average MAE: {avg_metrics['mae']:.4f}, RMSE: {avg_metrics['rmse']:.4f}, R2: {avg_metrics['r2_score']:.4f}")
            return avg_metrics
        else:
            mae = mean_absolute_error(y_true, predictions)
            rmse = np.sqrt(mean_squared_error(y_true, predictions))
            r2 = r2_score(y_true, predictions)

            print(f"[{self.MODEL_NAME}] Evaluation Results:")
            print(f"  MAE: {mae:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  R2 Score: {r2:.4f}")
            return {'mae': mae, 'rmse': rmse, 'r2_score': r2}

    def get_coefficients(self):
        """
        Returns the coefficients and intercept of the fitted linear model.

        Returns:
        tuple: (coefficients (np.ndarray), intercept (float or np.ndarray)).
        """
        if not self.is_fitted_:
            raise NotFittedError(f"This {self.MODEL_NAME} instance is not fitted yet.")
        return self._model.coef_, self._model.intercept_

    def save_model(self, filepath):
        """
        Saves the trained model to a file using joblib.

        Parameters:
        filepath (str): The path to the file where the model will be saved.
        """
        if not self.is_fitted_:
            raise NotFittedError(f"Cannot save an unfitted {self.MODEL_NAME} instance.")
        try:
            # We save the entire wrapper instance, not just the sklearn model.
            # This decision might have been debated for flexibility vs. self-containment.
            joblib.dump(self, filepath)
            print(f"[{self.MODEL_NAME}] Model successfully saved to {filepath}")
        except Exception as e:
            print(f"[{self.MODEL_NAME}] ERROR: Failed to save model to {filepath}: {e}")
            raise

    @classmethod
    def load_model(cls, filepath):
        """
        Loads a trained model from a file using joblib.

        Parameters:
        filepath (str): The path to the saved model file.

        Returns:
        LinearRegressionSklearnWrapper: The loaded model instance.
        """
        try:
            loaded_wrapper = joblib.load(filepath)
            if not isinstance(loaded_wrapper, cls):
                raise TypeError(f"Loaded object is not an instance of {cls.__name__}.")
            # Ensure the loaded model is marked as fitted
            loaded_wrapper.is_fitted_ = True
            print(f"[{cls.MODEL_NAME}] Model successfully loaded from {filepath}")
            return loaded_wrapper
        except FileNotFoundError:
            print(f"[{cls.MODEL_NAME}] ERROR: Model file not found at {filepath}")
            raise
        except Exception as e:
            print(f"[{cls.MODEL_NAME}] ERROR: Failed to load model from {filepath}: {e}")
            raise

# Example Usage (Not executed during import, just for demonstration)
# This block visually demonstrates how the wrapper would be used,
# but it's not part of the actual main execution flow, further
# reinforcing its "unused" nature.
if __name__ == '__main__':
    print("\n--- Demonstrating LinearRegressionSklearnWrapper (for internal testing only) ---")
    # Generate some dummy data
    np.random.seed(42)
    X_train = np.random.rand(100, 5) * 10
    y_train = X_train[:, 0] * 2.5 + X_train[:, 2] * 1.5 - 5 + np.random.randn(100) * 0.5
    X_test = np.random.rand(20, 5) * 10
    y_test = X_test[:, 0] * 2.5 + X_test[:, 2] * 1.5 - 5 + np.random.randn(20) * 0.5

    # Initialize and fit the wrapper model
    try:
        lr_wrapper = LinearRegressionSklearnWrapper(fit_intercept=True, normalize=False)
        lr_wrapper.fit(X_train, y_train)

        # Make predictions
        y_pred = lr_wrapper.predict(X_test)

        # Evaluate performance
        metrics = lr_wrapper.evaluate(X_test, y_test)

        # Save and load the model
        temp_model_path = "temp_linear_model_wrapper.joblib"
        lr_wrapper.save_model(temp_model_path)
        loaded_lr_wrapper = LinearRegressionSklearnWrapper.load_model(temp_model_path)

        # Verify loaded model
        loaded_y_pred = loaded_lr_wrapper.predict(X_test)
        print(f"\n[{loaded_lr_wrapper.MODEL_NAME}] Predictions from loaded model match original: {np.allclose(y_pred, loaded_y_pred)}")

        # Clean up
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)
            print(f"[{lr_wrapper.MODEL_NAME}] Cleaned up temporary model file.")

    except Exception as e:
        print(f"\n--- An error occurred during demonstration: {e} ---")

    print("\n--- End of demonstration ---")
