from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from data import DataPreprocessor
from svm import QSVM_Kernel
import pandas as pd
import argparse

class QSVM_Hybrid:
    def __init__(self, dataset='iris', param_grid=None):
        # Initialize data preprocessor and quantum SVM components
        self.data_preprocessor = DataPreprocessor(dataset=dataset)  # Pass dataset choice to DataPreprocessor
        self.param_grid = param_grid if param_grid else {
            'C': [0.1, 1, 10, 100],
            'reps': [1, 2, 3, 4],
            'feature_map_type': ['ZZ', 'Pauli']  # Adding feature map types for tuning
        }
        self.classical_svm = None
        self.best_params = None

    def load_and_preprocess_data(self):
        # Load and preprocess data
        X_train, X_test, y_train, y_test = self.data_preprocessor.get_processed_data()
        return X_train, X_test, y_train, y_test

    def print_data_summary(self, X_train, X_test, y_train, y_test):
        # Convert data to DataFrames for easier summary statistics
        X_train_df = pd.DataFrame(X_train)
        X_test_df = pd.DataFrame(X_test)
        y_train_df = pd.Series(y_train, name="y_train")
        y_test_df = pd.Series(y_test, name="y_test")
        pd.options.display.float_format = '{:.3f}'.format

        # Print summary statistics for features (X_train and X_test)
        print("\n--- Summary Statistics for Features (X_train) ---")
        print(X_train_df.describe())

        print("\n--- Summary Statistics for Features (X_test) ---")
        print(X_test_df.describe())

        # Print distribution of target labels
        print("\n--- Distribution of Target Labels (y_train) ---")
        print(y_train_df.value_counts())

        print("\n--- Distribution of Target Labels (y_test) ---")
        print(y_test_df.value_counts())

        # Optional: Show data shapes
        print("\nData Shapes:")
        print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
        print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")

    def tune_hyperparameters(self, X_train, y_train):
        best_score = 0
        best_params = {}
        
        # Iterate over all combinations of hyperparameters
        for reps in self.param_grid['reps']:
            for feature_map_type in self.param_grid['feature_map_type']:
                # Update the quantum kernel with the current values of reps and feature_map_type
                self.quantum_svm = QSVM_Kernel(
                    feature_dimension=X_train.shape[1], 
                    reps=reps, 
                    feature_map_type=feature_map_type
                )
                kernel_matrix_train = self.quantum_svm.get_kernel_matrix(X_train)

                # Define SVM with precomputed kernel and use GridSearchCV for hyperparameter tuning
                svc = SVC(kernel="precomputed")
                grid_search = GridSearchCV(svc, {'C': self.param_grid['C']}, cv=5, scoring='accuracy', n_jobs=-1)
                grid_search.fit(kernel_matrix_train, y_train)

                if grid_search.best_score_ > best_score:
                    best_score = grid_search.best_score_
                    best_params = {
                        'C': grid_search.best_params_['C'], 
                        'reps': reps, 
                        'feature_map_type': feature_map_type
                    }

        # Store the best parameters found by grid search
        self.best_params = best_params
        print(f"Best parameters found: {self.best_params}")
        print(f"Best cross-validation score: {best_score:.4f}")

    def train_with_best_params(self, X_train, y_train):
        # Update the quantum kernel with the best reps and feature_map_type values
        self.quantum_svm = QSVM_Kernel(
            feature_dimension=X_train.shape[1], 
            reps=self.best_params['reps'], 
            feature_map_type=self.best_params['feature_map_type']
        )
        kernel_matrix_train = self.quantum_svm.get_kernel_matrix(X_train)

        # Train a classical SVM with the best parameters from cross-validation
        self.classical_svm = SVC(kernel="precomputed", C=self.best_params['C'])
        self.classical_svm.fit(kernel_matrix_train, y_train)

    def predict(self, X_train, X_test):
        # Calculate the kernel matrix for the test data relative to the training data
        kernel_matrix_test = self.quantum_svm.quantum_kernel.evaluate(x_vec=X_test, y_vec=X_train)

        # Make predictions on the test data
        return self.classical_svm.predict(kernel_matrix_test)

    def evaluate(self, y_test, y_pred):
        # Evaluate the accuracy and print a classification report
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        return accuracy, report

    def run(self):
        # Step 1: Load and preprocess data
        X_train, X_test, y_train, y_test = self.load_and_preprocess_data()
        
        # Step 2: Print summary statistics
        print("Summary Statistics for the Dataset:")
        self.print_data_summary(X_train, X_test, y_train, y_test)

        # Step 3: Hyperparameter tuning with cross-validation
        print("Performing hyperparameter tuning for Quantum SVM...")
        self.tune_hyperparameters(X_train, y_train)
        
        # Step 4: Train with best hyperparameters on the full training set
        print("Training Quantum SVM with best hyperparameters on entire training data...")
        self.train_with_best_params(X_train, y_train)
        
        # Step 5: Predict and evaluate on test data
        print("Evaluating model...")
        y_pred = self.predict(X_train, X_test)
        accuracy, report = self.evaluate(y_test, y_pred)
        
        # Output results
        print(f"Final Model Accuracy: {accuracy * 100:.2f}%")
        print("\nClassification Report:\n", report)

class SVM:
    def __init__(self, dataset='iris', param_grid=None):
        # Initialize data preprocessor and set parameter grid for hyperparameter tuning
        self.data_preprocessor = DataPreprocessor(dataset=dataset)  # Pass dataset choice to DataPreprocessor
        self.param_grid = param_grid if param_grid else {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto'],
            'kernel': ['linear', 'rbf']
        }
        self.best_params = None
        self.classical_svm = None

    def load_and_preprocess_data(self):
        # Load and preprocess data
        X_train, X_test, y_train, y_test = self.data_preprocessor.get_processed_data()
        return X_train, X_test, y_train, y_test
    
    def tune_hyperparameters(self, X_train, y_train):
        # Use GridSearchCV for hyperparameter tuning
        svc = SVC()
        grid_search = GridSearchCV(svc, self.param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Store the best parameters found by grid search
        self.best_params = grid_search.best_params_
        print(f"Best parameters found for Classical SVM: {self.best_params}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

    def train_with_best_params(self, X_train, y_train):
        # Train a classical SVM with the best parameters from cross-validation
        self.classical_svm = SVC(**self.best_params)
        self.classical_svm.fit(X_train, y_train)

    def predict(self, X_test):
        # Make predictions on the test data
        return self.classical_svm.predict(X_test)

    def evaluate(self, y_test, y_pred):
        # Evaluate the accuracy and print a classification report
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        return accuracy, report

    def run(self):
        # Step 1: Load and preprocess data
        X_train, X_test, y_train, y_test = self.load_and_preprocess_data()
        
        # Step 3: Hyperparameter tuning with cross-validation
        print("Performing hyperparameter tuning for Classical SVM...")
        self.tune_hyperparameters(X_train, y_train)
        
        # Step 4: Train with best hyperparameters on the full training set
        print("Training Classical SVM with best hyperparameters on entire training data...")
        self.train_with_best_params(X_train, y_train)
        
        # Step 5: Predict and evaluate on test data
        print("Evaluating Classical SVM model...")
        y_pred = self.predict(X_test)
        accuracy, report = self.evaluate(y_test, y_pred)
        
        # Output results
        print(f"Classical SVM Final Model Accuracy: {accuracy * 100:.2f}%")
        print("\nClassical SVM Classification Report:\n", report)

if __name__ == "__main__": 
    choice = ""
    while not choice:
        choice = input("Choose iris or adhoc: ")
        if choice != "iris" and choice != "adhoc":
            print("Incorrect input: Choose iris or adhoc")
            choice = ""

    print(f"{choice} chosen. Running comparison between QVSM Hybrid and Classic SVM")
    quantum = QSVM_Hybrid(dataset=choice)
    quantum.run()
    classical = SVM(dataset=choice)
    classical.run()
