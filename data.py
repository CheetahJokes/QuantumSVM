from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from qiskit_machine_learning.datasets import ad_hoc_data
import numpy as np

class DataPreprocessor:
    def __init__(self, test_size=0.2, random_state=42, dataset='iris'):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.dataset = dataset  # 'iris' or 'adhoc'
        
    def load_data(self):
        if self.dataset == 'iris':
            # Load the Iris dataset
            iris = load_iris()
            X = iris.data
            y = iris.target

            # Reduce to binary classification (e.g., class 0 and class 1)
            binary_filter = y < 2  # Only take class 0 and 1
            X = X[binary_filter]
            y = y[binary_filter]
        
        elif self.dataset == 'adhoc':
            # Load the ad-hoc dataset from Qiskit Machine Learning
            train_features, train_labels, test_features, test_labels = ad_hoc_data(
                training_size=80, test_size=20, n=2, gap=0.3, plot_data=False, one_hot=False
            )

            # Combine training and test data for consistent processing
            X = np.concatenate([train_features, test_features])
            y = np.concatenate([train_labels, test_labels])
        
        else:
            raise ValueError("Invalid dataset choice. Choose 'iris' or 'adhoc'.")
        
        return X, y

    def preprocess_data(self, X, y):
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        # Normalize the features using standard scaling
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test

    def get_processed_data(self):
        # Load and preprocess data in one step
        X, y = self.load_data()
        return self.preprocess_data(X, y)
