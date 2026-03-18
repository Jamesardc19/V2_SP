from sklearn.base import BaseEstimator, ClassifierMixin

class TabNetClassifierWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper class for TabNetClassifier to make it compatible with sklearn's calibration methods.
    """
    def __init__(self, tabnet_model):
        self.tabnet_model = tabnet_model
        
    def fit(self, X, y):
        # TabNet is already fitted, so just return self
        return self
        
    def predict(self, X):
        # Convert to numpy if needed
        X_np = X.values if hasattr(X, 'values') else X
        return self.tabnet_model.predict(X_np)
    
    def predict_proba(self, X):
        # Convert to numpy if needed
        X_np = X.values if hasattr(X, 'values') else X
        return self.tabnet_model.predict_proba(X_np)
