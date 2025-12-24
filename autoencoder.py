import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Callable

class Autoencoder:
    def __init__(self,
        input_dim: int,
        encoder_dims: List[int], # hidden encoder layers, atleast 3 layers
        bottleneck_dim: int,
        decoder_dims: List[int], # hidden decoder layers, atleast 3 layers
        activation: str = 'relu', # ('relu', 'sigmoid', 'tanh')
        learning_rate: float = 0.001,
        l2_lambda: float = 0.0001 # L2 regularization parameter. A higher value forces the weights to be smaller.
    ):
        assert len(encoder_dims) >= 3, "Encoder must have at least 3 hidden layers"
        assert len(decoder_dims) >= 3, "Decoder must have at least 3 hidden layers"

        self.input_dim = input_dim
        self.encoder_dims = encoder_dims
        self.bottleneck_dim = bottleneck_dim
        self.decoder_dims = decoder_dims
        self.activation = activation
        self.learning_rate = learning_rate
        self.initial_lr = learning_rate
        self.l2_lambda = l2_lambda

        # Initialize weights and biases
        self.params = {}
        self._initialize_parameters()

        # Cache for storing forward pass values (needed for backprop)
        self.cache = {}

    def _initialize_parameters(self):
        """Initialize weights using He initialization for ReLU, Xavier for others"""
        np.random.seed(42)

        # Encoder weights
        encoder_layers = [self.input_dim] + self.encoder_dims + [self.bottleneck_dim]
        for i in range(len(encoder_layers) - 1):
            # He initialization for ReLU, Xavier for others
            if self.activation == 'relu':
                scale = np.sqrt(2.0 / encoder_layers[i]) # ReLU kills half the neurons (negatives → 0), so we need bigger weights to compensate and keep variance through the layers
            else:
                scale = np.sqrt(1.0 / encoder_layers[i])

            # now set the initial weights of each layer multiplied by the scaling factor
            # scalling factor is important for balanced signals for learning
            self.params[f'W_enc_{i}'] = np.random.randn(
                encoder_layers[i], encoder_layers[i+1]
            ) * scale
            self.params[f'b_enc_{i}'] = np.zeros((1, encoder_layers[i+1]))

        # Decoder weights
        decoder_layers = [self.bottleneck_dim] + self.decoder_dims + [self.input_dim]
        for i in range(len(decoder_layers) - 1):
            if self.activation == 'relu' and i < len(decoder_layers) - 2:
                scale = np.sqrt(2.0 / decoder_layers[i])
            else:
                scale = np.sqrt(1.0 / decoder_layers[i])

            self.params[f'W_dec_{i}'] = np.random.randn(
                decoder_layers[i], decoder_layers[i+1]
            ) * scale
            self.params[f'b_dec_{i}'] = np.zeros((1, decoder_layers[i+1]))

    # ============= Activation Functions =============

    def _relu(self, z):
        return np.maximum(0, z)

    def _relu_derivative(self, z):
        return (z > 0).astype(float)

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def _sigmoid_derivative(self, z):
        s = self._sigmoid(z)
        return s * (1 - s)

    def _tanh(self, z):
        return np.tanh(z)

    def _tanh_derivative(self, z):
        return 1 - np.tanh(z) ** 2

    def _activate(self, z):
        if self.activation == 'relu':
            return self._relu(z)
        elif self.activation == 'sigmoid':
            return self._sigmoid(z)
        elif self.activation == 'tanh':
            return self._tanh(z)

    def _activate_derivative(self, z):
        if self.activation == 'relu':
            return self._relu_derivative(z)
        elif self.activation == 'sigmoid':
            return self._sigmoid_derivative(z)
        elif self.activation == 'tanh':
            return self._tanh_derivative(z)


    def forward(self, X):
        """
        Forward pass through the autoencoder

        Parameters:
        -----------
        X : ndarray of shape (batch_size, input_dim)
            Input data

        Returns:
        --------
        reconstruction : ndarray of shape (batch_size, input_dim)
            Reconstructed data
        """
        self.cache = {}
        self.cache['A_0'] = X

        # Encoder forward pass
        A = X
        num_encoder_layers = len(self.encoder_dims) + 1  # +1 for bottleneck

        for i in range(num_encoder_layers):
            W = self.params[f'W_enc_{i}']
            b = self.params[f'b_enc_{i}']

            Z = np.dot(A, W) + b
            self.cache[f'Z_enc_{i}'] = Z

            # Apply activation (for all encoder layers including bottleneck)
            A = self._activate(Z)
            self.cache[f'A_enc_{i}'] = A

        # Bottleneck is the last encoder activation
        self.cache['bottleneck'] = A

        # Decoder forward pass
        num_decoder_layers = len(self.decoder_dims) + 1  # +1 for output

        for i in range(num_decoder_layers):
            W = self.params[f'W_dec_{i}']
            b = self.params[f'b_dec_{i}']

            Z = np.dot(A, W) + b
            self.cache[f'Z_dec_{i}'] = Z

            # Apply activation for hidden layers, sigmoid for output layer
            if i < num_decoder_layers - 1:
                A = self._activate(Z)
            else:
                # Output layer - use sigmoid to bound output to [0, 1]
                A = self._sigmoid(Z)

            self.cache[f'A_dec_{i}'] = A

        return A

    # ============= Loss Function =============

    def compute_loss(self, X, X_reconstructed):
        """
        Compute Mean Squared Error loss with L2 regularization

        Parameters:
        -----------
        X : ndarray of shape (batch_size, input_dim)
            Original input
        X_reconstructed : ndarray of shape (batch_size, input_dim)
            Reconstructed output

        Returns:
        --------
        loss : float
            Total loss (MSE + L2 regularization)
        """
        m = X.shape[0]

        # Mean Squared Error
        mse_loss = np.mean((X - X_reconstructed) ** 2)

        # L2 Regularization
        l2_loss = 0
        for key in self.params:
            if key.startswith('W_'):
                l2_loss += np.sum(self.params[key] ** 2)

        l2_loss = (self.l2_lambda / (2 * m)) * l2_loss

        return mse_loss + l2_loss
    def evaluate_mse(self, X):
      """
      Returns only the Mean Squared Error (no L2 penalty)
      """
      # 1. Get the reconstruction
      X_reconstructed = self.reconstruct(X)

      # 2. Calculate MSE: Average of (Actual - Predicted)^2
      mse = np.mean((X - X_reconstructed) ** 2)

      return mse

    # ============= Backpropagation =============

    def backward(self, X, X_reconstructed):
        """
        Backward pass - compute gradients using backpropagation

        Parameters:
        -----------
        X : ndarray of shape (batch_size, input_dim)
            Original input
        X_reconstructed : ndarray of shape (batch_size, input_dim)
            Reconstructed output

        Returns:
        --------
        gradients : dict
            Dictionary containing gradients for all parameters
        """
        m = X.shape[0]
        gradients = {}

        # ===== DECODER BACKPROPAGATION =====

        # Output layer gradient (derivative of MSE loss)
        # Loss = (1/m) * sum((y_pred - y_true)^2)
        # dL/dy_pred = (2/m) * (y_pred - y_true)
        dA = (2 / m) * (X_reconstructed - X)

        num_decoder_layers = len(self.decoder_dims) + 1

        for i in range(num_decoder_layers - 1, -1, -1):
            # Get cached values
            Z = self.cache[f'Z_dec_{i}']

            if i > 0:
                A_prev = self.cache[f'A_dec_{i-1}']
            else:
                # First decoder layer uses bottleneck as input
                A_prev = self.cache['bottleneck']

            # Apply activation derivative
            if i == num_decoder_layers - 1:
                # Output layer uses sigmoid
                dZ = dA * self._sigmoid_derivative(Z)
            else:
                dZ = dA * self._activate_derivative(Z)

            # Compute gradients
            W = self.params[f'W_dec_{i}']

            gradients[f'W_dec_{i}'] = np.dot(A_prev.T, dZ) + (self.l2_lambda / m) * W
            gradients[f'b_dec_{i}'] = np.sum(dZ, axis=0, keepdims=True)

            # Propagate gradient to previous layer
            dA = np.dot(dZ, W.T)

        # ===== ENCODER BACKPROPAGATION =====

        # dA now contains gradient flowing into bottleneck
        num_encoder_layers = len(self.encoder_dims) + 1

        for i in range(num_encoder_layers - 1, -1, -1):
            # Get cached values
            Z = self.cache[f'Z_enc_{i}']

            if i > 0:
                A_prev = self.cache[f'A_enc_{i-1}']
            else:
                # First encoder layer uses input
                A_prev = self.cache['A_0']

            # Apply activation derivative
            dZ = dA * self._activate_derivative(Z)

            # Compute gradients
            W = self.params[f'W_enc_{i}']

            gradients[f'W_enc_{i}'] = np.dot(A_prev.T, dZ) + (self.l2_lambda / m) * W
            gradients[f'b_enc_{i}'] = np.sum(dZ, axis=0, keepdims=True)

            # Propagate gradient to previous layer
            dA = np.dot(dZ, W.T)

        return gradients

    # ============= Parameter Updates =============

    def update_parameters(self, gradients):
        """Update parameters using computed gradients"""
        for key in self.params:
            self.params[key] -= self.learning_rate * gradients[key]  # Wnew ​= Wold ​− ( lambda × ∇L )

    # ============= Learning Rate Scheduling =============

    def update_learning_rate(self, epoch, schedule='step'):
        """
        Update learning rate based on schedule
        Its job is to decrease the learning rate as training progresses to prevent overshooting.
        Parameters:
        -----------
        epoch : int
            Current epoch number
        schedule : str
            Type of schedule ('step', 'exponential', 'cosine')
        """
        if schedule == 'step':
            # Reduce LR by factor of 10 every 50 epochs
            if epoch > 0 and epoch % 50 == 0:
                self.learning_rate *= 0.1

        elif schedule == 'exponential':
            # Exponential decay: lr = initial_lr * exp(-decay_rate * epoch)
            decay_rate = 0.01
            self.learning_rate = self.initial_lr * np.exp(-decay_rate * epoch)

        elif schedule == 'cosine':
            # Cosine annealing
            min_lr = self.initial_lr * 0.01
            max_epochs = 100
            self.learning_rate = min_lr + 0.5 * (self.initial_lr - min_lr) * (
                1 + np.cos(np.pi * epoch / max_epochs)
            )

    # ============= Training =============

    def fit(self,
        X_train,
        epochs=100,
        batch_size=32,
        lr_schedule='step',
        verbose=True
    ):
        """
        Train the autoencoder using mini-batch gradient descent

        Parameters:
        -----------
        X_train : ndarray of shape (n_samples, input_dim)
            Training data
        epochs : int
            Number of training epochs
        batch_size : int
            Size of mini-batches
        lr_schedule : str
            Learning rate schedule type
        verbose : bool
            Whether to print training progress

        Returns:
        --------
        history : dict
            Dictionary containing training history (losses)
        """
        n_samples = X_train.shape[0]
        history = {'total_loss': [],
                   'mse_loss': []}

        for epoch in range(epochs):
            # Shuffle data to avoid overfitting
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]

            epoch_total_losses = []
            epoch_mse_only = []

            # Mini-batch gradient descent
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]

                # Forward pass
                X_reconstructed = self.forward(X_batch)

                # Calculate Pure MSE for reporting
                pure_mse = np.mean((X_batch - X_reconstructed) ** 2)
                epoch_mse_only.append(pure_mse)

                # Compute Total Loss for backprop (MSE + L2)
                loss = self.compute_loss(X_batch, X_reconstructed)
                epoch_total_losses.append(loss)

                # Backward pass
                gradients = self.backward(X_batch, X_reconstructed)

                # Update parameters
                self.update_parameters(gradients)

            # Average loss for epoch
            avg_loss = np.mean(epoch_total_losses)
            avg_mse = np.mean(epoch_mse_only)
            history['total_loss'].append(avg_loss)
            history['mse_loss'].append(avg_mse)

            # Update learning rate
            self.update_learning_rate(epoch, schedule=lr_schedule)

            # Print progress
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch} - Total Loss: {avg_loss:.4f} - Reconstruction MSE: {avg_mse:.6f}")
         

        return history

    # ============= Utility Methods =============

    def encode(self, X):
        """Encode input to bottleneck representation"""
        self.forward(X)
        return self.cache['bottleneck']

    def decode(self, Z):
        """Decode bottleneck representation to output"""
        A = Z
        num_decoder_layers = len(self.decoder_dims) + 1

        for i in range(num_decoder_layers):
            W = self.params[f'W_dec_{i}']
            b = self.params[f'b_dec_{i}']
            Z_dec = np.dot(A, W) + b

            if i < num_decoder_layers - 1:
                A = self._activate(Z_dec)
            else:
                A = self._sigmoid(Z_dec)

        return A

    def reconstruct(self, X):
        """Full reconstruction: encode then decode"""
        return self.forward(X)