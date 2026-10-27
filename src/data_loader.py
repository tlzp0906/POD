
#### 4. `src/data_loader.py`

```python
import pandas as pd
from sklearn.model_selection import train_test_split

def load_excel_data(file_path: str, sheet_name: str = 0, header: int = 0) -> pd.DataFrame:
    """
    Loads data from an Excel file.

    Args:
        file_path (str): Path to the Excel file.
        sheet_name (str or int): Sheet name or sheet index. Defaults to 0 (first sheet).
        header (int): Row number to use as the column names. Defaults to 0 (first row).

    Returns:
        pd.DataFrame: Loaded data.
    """
    try:
        data = pd.read_excel(file_path, sheet_name=sheet_name, header=header)
        print(f"Successfully loaded data from {file_path}, sheet: {sheet_name}")
        print(f"Data shape: {data.shape}")
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading Excel data from {file_path}: {e}")
        return pd.DataFrame()

def load_and_split_mlr_data(train_file: str, test_file: str, feature_cols: list, target_col: str):
    """
    Loads pre-split train/test data for MLR model.

    Args:
        train_file (str): Path to the training Excel file.
        test_file (str): Path to the testing Excel file.
        feature_cols (list): List of column names or indices for features.
        target_col (str): Column name or index for the target variable.

    Returns:
        tuple: (X_train, y_train, X_test, y_test, feature_names)
    """
    train_df = load_excel_data(train_file)
    test_df = load_excel_data(test_file)

    if train_df.empty or test_df.empty:
        return None, None, None, None, None

    # Ensure feature_cols are column names for consistency
    if all(isinstance(col, int) for col in feature_cols):
        feature_names = train_df.columns[feature_cols].tolist()
    else:
        feature_names = feature_cols
    
    # Extract features and target
    X_train = train_df[feature_names].values
    y_train = train_df[target_col].values
    X_test = test_df[feature_names].values
    y_test = test_df[target_col].values
    
    print(f"MLR Data loaded. X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"MLR Data loaded. X_test: {X_test.shape}, y_test: {y_test.shape}")

    return X_train, y_train, X_test, y_test, feature_names

def load_and_split_data(file_path: str, test_size: float = 0.3, random_state: int = 42,
                        target_col_index: int = -1, feature_col_indices: list = None,
                        sheet_name: str = 0) -> tuple:
    """
    Loads data from an Excel file and splits it into training and testing sets.

    Args:
        file_path (str): Path to the Excel file.
        test_size (float): Proportion of the dataset to include in the test split. Defaults to 0.3 (30%).
        random_state (int): Seed for random number generator.
        target_col_index (int): Column index of the target variable. Defaults to -1 (last column).
        feature_col_indices (list): List of column indices for features. If None, all columns except target_col_index are used.
        sheet_name (str or int): Sheet name or sheet index.

    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names)
    """
    data_df = load_excel_data(file_path, sheet_name=sheet_name)
    if data_df.empty:
        return None, None, None, None, None

    if feature_col_indices is None:
        # Use all columns except the target column
        X = data_df.drop(columns=data_df.columns[target_col_index])
        feature_names = X.columns.tolist()
        X = X.values
    else:
        # Use specified feature columns
        X = data_df.iloc[:, feature_col_indices].values
        feature_names = data_df.columns[feature_col_indices].tolist()

    y = data_df.iloc[:, target_col_index].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"Full Data loaded and split. X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

    return X_train, X_test, y_train, y_test, feature_names
