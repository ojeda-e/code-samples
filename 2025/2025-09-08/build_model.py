"""
This module contains code for building two machine learning models using PyTorch Lightning:
a Random Forest model and a Neural Network model.
The models are trained on imbalanced data generated with SynthBioData.

Notes
----------
- Data generation is handled in the generate_data.py module.
- PyTorch Lightning is used for modular model training and evaluation.
- This script focuses on model definition and training.
"""

import logging
from typing import Tuple
import abc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


class DrugDiscoveryDataset(Dataset):
    """
    PyTorch Dataset for drug discovery features and labels.

    Parameters
    ----------
    features : torch.Tensor
        Feature tensor of shape (n_samples, n_features).
    labels : torch.Tensor
        Label tensor of shape (n_samples,) with binary labels.
    """
    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.

        Returns
        -------
        int
            Number of samples.
        """
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve a single sample and label by index.

        Parameters
        ----------
        idx : int
            Index of the sample.

        Returns
        -------
        tuple of torch.Tensor
            (feature, label) pair.
        """
        return self.features[idx], self.labels[idx]


class BaseModel(pl.LightningModule, abc.ABC):
    """Abstract base class for PyTorch Lightning drug discovery models."""
    
    def __init__(self, n_features: int):
        super().__init__()
        self.n_features = n_features
        self.save_hyperparameters()

        self.predictions = {
            'train': {'preds': [], 'targets': []},
            'val': {'preds': [], 'targets': []},
            'test': {'preds': [], 'targets': []}
        }

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Parameters
        ----------
        x : torch.Tensor
            Input features.

        Returns
        -------
        torch.Tensor
            Model output.
        """
        pass

    @abc.abstractmethod
    def compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss for a batch.

        Parameters
        ----------
        y_pred : torch.Tensor
            Model predictions.
        y_true : torch.Tensor
            Ground truth labels.

        Returns
        -------
        torch.Tensor
            Loss value.
        """
        pass

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Perform a single training step.

        Parameters
        ----------
        batch : tuple of torch.Tensor
            Batch of (features, labels).
        batch_idx : int
            Batch index.

        Returns
        -------
        torch.Tensor
            Loss value.
        """
        x, y = batch
        y_pred = self(x)
        loss = self.compute_loss(y_pred, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        self.predictions['train']['preds'].append(y_pred.detach())
        self.predictions['train']['targets'].append(y.detach())
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Perform a single validation step.

        Parameters
        ----------
        batch : tuple of torch.Tensor
            Batch of (features, labels).
        batch_idx : int
            Batch index.

        Returns
        -------
        torch.Tensor
            Loss value.
        """
        x, y = batch
        y_pred = self(x)
        loss = self.compute_loss(y_pred, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        self.predictions['val']['preds'].append(y_pred.detach())
        self.predictions['val']['targets'].append(y.detach())
        return loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Perform a single test step.

        Parameters
        ----------
        batch : tuple of torch.Tensor
            Batch of (features, labels).
        batch_idx : int
            Batch index.

        Returns
        -------
        torch.Tensor
            Loss value.
        """
        x, y = batch
        y_pred = self(x)
        loss = self.compute_loss(y_pred, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        
        # Store predictions
        self.predictions['test']['preds'].append(y_pred.detach())
        self.predictions['test']['targets'].append(y.detach())
        return loss

    def on_train_epoch_end(self):
        """Get final predictions for the epoch."""
        if self.predictions['train']['preds']:
            preds = torch.cat(self.predictions['train']['preds'])
            targets = torch.cat(self.predictions['train']['targets'])
            self.predictions['train']['final_preds'] = preds.cpu().numpy()
            self.predictions['train']['final_targets'] = targets.cpu().numpy()
            self.predictions['train']['preds'] = []
            self.predictions['train']['targets'] = []

    def on_validation_epoch_end(self):
        """Get final predictions for the epoch."""
        if self.predictions['val']['preds']:
            preds = torch.cat(self.predictions['val']['preds'])
            targets = torch.cat(self.predictions['val']['targets'])
            self.predictions['val']['final_preds'] = preds.cpu().numpy()
            self.predictions['val']['final_targets'] = targets.cpu().numpy()
            self.predictions['val']['preds'] = []
            self.predictions['val']['targets'] = []

    def on_test_epoch_end(self):
        """Get final predictions for the epoch."""
        if self.predictions['test']['preds']:
            preds = torch.cat(self.predictions['test']['preds'])
            targets = torch.cat(self.predictions['test']['targets'])
            self.predictions['test']['final_preds'] = preds.cpu().numpy()
            self.predictions['test']['final_targets'] = targets.cpu().numpy()
            self.predictions['test']['preds'] = []
            self.predictions['test']['targets'] = []

    def configure_optimizers(self):
        """Configure the optimizer."""
        return torch.optim.Adam(self.parameters(), lr=0.001)




class RandomForestModel(BaseModel):
    """PyTorch Lightning wrapper for a Random Forest model."""
    
    def __init__(self, n_features: int):
        super().__init__(n_features)
        self.sklearn_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
        self.classifier = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using a linear layer.

        Parameters
        ----------
        x : torch.Tensor
            Input features.

        Returns
        -------
        torch.Tensor
            Output probabilities.
        """
        return torch.sigmoid(self.classifier(x))

    def compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute binary cross-entropy loss.

        Parameters
        ----------
        y_pred : torch.Tensor
            Model predictions.
        y_true : torch.Tensor
            Ground truth labels.

        Returns
        -------
        torch.Tensor
            Loss value.
        """
        return F.binary_cross_entropy(y_pred.squeeze(), y_true.float())

    def fit_sklearn_model(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the underlying sklearn Random Forest model.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Target vector.
        """
        self.sklearn_model.fit(X, y)
        feature_importance = self.sklearn_model.feature_importances_
        with torch.no_grad():
            self.classifier.weight.data = torch.tensor(feature_importance).unsqueeze(0)
            self.classifier.bias.data = torch.tensor([0.0])

    def predict_sklearn(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict using the sklearn Random Forest model.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.

        Returns
        -------
        tuple of np.ndarray
            (predicted labels, predicted probabilities)
        """
        y_pred = self.sklearn_model.predict(X)
        y_pred_proba = self.sklearn_model.predict_proba(X)[:, 1]
        return y_pred, y_pred_proba


class NeuralNetworkModel(BaseModel):
    """PyTorch Lightning neural network model for drug discovery."""
    
    def __init__(self, n_features: int):
        super().__init__(n_features)
        self.network = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the neural network.

        Parameters
        ----------
        x : torch.Tensor
            Input features.

        Returns
        -------
        torch.Tensor
            Output probabilities.
        """
        return torch.sigmoid(self.network(x))

    def compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute binary cross-entropy loss."""
        return F.binary_cross_entropy(y_pred.squeeze(), y_true.float())


def prepare_data(df) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Prepare features and targets for PyTorch Lightning training."""
    # Check if required column exists - this comes from the generate_data.py module
    if 'binds_target' not in df.columns:
        raise ValueError("Required column 'binds_target' not found in DataFrame")
    
    # Get feature columns and prepare features and targets
    feature_cols = [col for col in df.columns if col != 'binds_target']
    if not feature_cols:
        raise ValueError("No feature columns found in DataFrame")
    
    logger.info(f"Using {len(feature_cols)} feature columns")
    
    X = df.select(feature_cols)
    y = df['binds_target'].to_numpy()
    
    categorical_cols = []
    for col in ['target_family', 'formal_charge']:
        if col in X.columns:
            categorical_cols.append(col)
    
    if categorical_cols:
        logger.info(f"Encoding categorical columns: {categorical_cols}")
        X_encoded = X.to_dummies(columns=categorical_cols)
    else:
        X_encoded = X
    
    # Convert to tensors
    X_tensor = torch.tensor(X_encoded.to_numpy(), dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    logger.info(f"Data prepared: {X_tensor.shape[0]} samples, {X_tensor.shape[1]} features")
    
    return X_tensor, y_tensor, X_encoded.shape[1]


def create_data_loaders(X: torch.Tensor, y: torch.Tensor) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test data loaders."""
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y.numpy()
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp.numpy()
    )
    
    train_dataset = DrugDiscoveryDataset(X_train, y_train)
    val_dataset = DrugDiscoveryDataset(X_val, y_val)
    test_dataset = DrugDiscoveryDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    logger.info(f"Training set: {len(train_dataset):,} samples")
    logger.info(f"Validation set: {len(val_dataset):,} samples")
    logger.info(f"Test set: {len(test_dataset):,} samples")
    
    return train_loader, val_loader, test_loader


def train_model(df, model_type: str = "neural_network"):
    """Train a PyTorch Lightning model and return predictions."""
    logger.info("Preparing data for training...")
    X, y, n_features = prepare_data(df)
    train_loader, val_loader, test_loader = create_data_loaders(X, y)
    
    if model_type == "neural_network":
        logger.info("Initializing Neural Network model...")
        model = NeuralNetworkModel(n_features)
    elif model_type == "random_forest":
        logger.info("Initializing Random Forest model...")
        model = RandomForestModel(n_features)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    callbacks = [
        ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1),
        EarlyStopping(monitor='val_loss', mode='min', patience=10)
    ]
    
    trainer = pl.Trainer(
        max_epochs=30,
        callbacks=callbacks,
        accelerator="cpu",
        devices=1,
        enable_progress_bar=True,
        deterministic=True
    )
    
    logger.info(f"Training {model_type} model...")
    trainer.fit(model, train_loader, val_loader)
    logger.info("Evaluating model on test set...")
    trainer.test(model, test_loader)
    
    return {
        'model_type': model_type,
        'predictions': model.predictions
    }
