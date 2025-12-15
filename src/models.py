from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
import joblib

class AcademicModel:
    def __init__(self):
        self.classification_model = None
        self.clustering_model = None
    
    def train_classification(self, X, y, model_type='decision_tree', **params):
        """Train classification model"""
        if model_type == 'decision_tree':
            self.classification_model = DecisionTreeClassifier(**params)
        
        self.classification_model.fit(X, y)
        return self.classification_model
    
    def train_clustering(self, X, n_clusters=3, **params):
        """Train clustering model"""
        self.clustering_model = KMeans(n_clusters=n_clusters, **params)
        clusters = self.clustering_model.fit_predict(X)
        return clusters
    
    def save_model(self, filepath):
        """Save trained model"""
        if self.classification_model:
            joblib.dump(self.classification_model, filepath)
    
    def load_model(self, filepath):
        """Load saved model"""
        self.classification_model = joblib.load(filepath)
        return self.classification_model
    
print("AcademicModel module loaded successfully.")