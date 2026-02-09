import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, compute_model_metrics, inference, save_model, load_model
from ml.data import process_data
import os
import tempfile


@pytest.fixture
def sample_data():
    """
    Fixture to create sample training data for testing
    """
    # Create a small sample dataset
    data = pd.DataFrame({
        'age': [39, 50, 38, 53, 28],
        'workclass': ['State-gov', 'Self-emp-not-inc', 'Private', 'Private', 'Private'],
        'education': ['Bachelors', 'Bachelors', 'HS-grad', '11th', 'Bachelors'],
        'marital-status': ['Never-married', 'Married-civ-spouse', 'Divorced', 'Married-civ-spouse', 'Married-civ-spouse'],
        'occupation': ['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners', 'Handlers-cleaners', 'Prof-specialty'],
        'relationship': ['Not-in-family', 'Husband', 'Not-in-family', 'Husband', 'Wife'],
        'race': ['White', 'White', 'White', 'Black', 'Black'],
        'sex': ['Male', 'Male', 'Male', 'Male', 'Female'],
        'native-country': ['United-States', 'United-States', 'United-States', 'United-States', 'Cuba'],
        'salary': ['<=50K', '<=50K', '<=50K', '<=50K', '<=50K']
    })
    
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    
    return data, cat_features


def test_train_model_returns_correct_type(sample_data):
    """
    Test that train_model returns a RandomForestClassifier model
    """
    data, cat_features = sample_data
    X, y, encoder, lb = process_data(
        data, categorical_features=cat_features, label="salary", training=True
    )
    
    model = train_model(X, y)
    
    # Check that the model is of the correct type
    assert isinstance(model, RandomForestClassifier), "Model should be a RandomForestClassifier"
    # Check that the model has been fitted
    assert hasattr(model, 'estimators_'), "Model should be fitted"


def test_compute_model_metrics_returns_expected_values():
    """
    Test that compute_model_metrics returns values in the expected range [0, 1]
    """
    # Create sample predictions and labels
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 0, 1])
    
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    
    # Check that all metrics are between 0 and 1
    assert 0 <= precision <= 1, "Precision should be between 0 and 1"
    assert 0 <= recall <= 1, "Recall should be between 0 and 1"
    assert 0 <= fbeta <= 1, "F1 score should be between 0 and 1"
    
    # Check that metrics are floats
    assert isinstance(precision, (float, np.floating)), "Precision should be a float"
    assert isinstance(recall, (float, np.floating)), "Recall should be a float"
    assert isinstance(fbeta, (float, np.floating)), "F1 should be a float"


def test_inference_returns_predictions(sample_data):
    """
    Test that inference function returns predictions of correct shape and type
    """
    data, cat_features = sample_data
    X, y, encoder, lb = process_data(
        data, categorical_features=cat_features, label="salary", training=True
    )
    
    # Train a simple model
    model = train_model(X, y)
    
    # Run inference
    preds = inference(model, X)
    
    # Check that predictions have the correct shape
    assert len(preds) == len(y), "Predictions should have same length as input"
    
    # Check that predictions are binary (0 or 1)
    assert set(preds).issubset({0, 1}), "Predictions should be binary (0 or 1)"
    
    # Check that predictions are numpy array
    assert isinstance(preds, np.ndarray), "Predictions should be a numpy array"


def test_save_and_load_model(sample_data):
    """
    Test that save_model and load_model work correctly
    """
    data, cat_features = sample_data
    X, y, encoder, lb = process_data(
        data, categorical_features=cat_features, label="salary", training=True
    )
    
    # Train a model
    model = train_model(X, y)
    
    # Save the model to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        save_model(model, tmp_path)
        
        # Check that file was created
        assert os.path.exists(tmp_path), "Model file should be created"
        
        # Load the model
        loaded_model = load_model(tmp_path)
        
        # Check that loaded model is of correct type
        assert isinstance(loaded_model, RandomForestClassifier), "Loaded model should be RandomForestClassifier"
        
        # Check that loaded model makes same predictions
        original_preds = inference(model, X)
        loaded_preds = inference(loaded_model, X)
        assert np.array_equal(original_preds, loaded_preds), "Loaded model should make same predictions"
        
    finally:
        # Clean up
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_data_shape_after_processing(sample_data):
    """
    Test that processed data has the expected shape
    """
    data, cat_features = sample_data
    
    X, y, encoder, lb = process_data(
        data, categorical_features=cat_features, label="salary", training=True
    )
    
    # Check that X and y have the same number of samples
    assert X.shape[0] == len(y), "X and y should have same number of samples"
    
    # Check that X has more features than original (due to one-hot encoding)
    assert X.shape[1] > len(data.columns) - 1, "Processed X should have more features due to encoding"
    
    # Check that y is binary
    assert set(y).issubset({0, 1}), "Labels should be binary after processing"
