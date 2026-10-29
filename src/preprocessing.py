"""
EduGuard AI — Preprocessing Pipeline Builder
Builds scikit-learn ColumnTransformer and Pipeline ensuring zero leakage between train and test sets.
"""
from typing import List, Tuple
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str]
) -> ColumnTransformer:
    """
    Construct a leakage-free preprocessing ColumnTransformer:
    - Numerical: median imputation + standard scaling
    - Categorical: most frequent imputation + one-hot encoding (handle_unknown='ignore')
    """
    transformers = []

    if numeric_features:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("num", numeric_pipeline, numeric_features))

    if categorical_features:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(("cat", categorical_pipeline, categorical_features))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )
    return preprocessor

def get_feature_names_out(preprocessor: ColumnTransformer, numeric_features: List[str], categorical_features: List[str]) -> List[str]:
    """Retrieve transformed feature names safely across scikit-learn versions."""
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback manual reconstruction
        names = list(numeric_features)
        if categorical_features and "cat" in preprocessor.named_transformers_:
            ohe = preprocessor.named_transformers_["cat"].named_steps.get("onehot")
            if hasattr(ohe, "get_feature_names_out"):
                names.extend(list(ohe.get_feature_names_out(categorical_features)))
            else:
                names.extend(categorical_features)
        return names
