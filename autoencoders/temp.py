import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def sigmoid_derivative(x):
    return x * (1 - x)


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


class AutoencoderScratch:
    def __init__(self, input_size, hidden_size, learning_rate=0.01):
        """
        Simple 3-layer autoencoder: Input -> Hidden -> Output
        """
        # Initialize weights with He initialization
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, input_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, input_size))

        self.learning_rate = learning_rate

    def forward(self, X):
        """Forward pass through encoder and decoder"""
        # Encoder
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = relu(self.z1)

        # Decoder
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)

        return self.a2

    def backward(self, X, output):
        """Backpropagation to compute gradients"""
        m = X.shape[0]

        # Output layer gradients
        da2 = output - X  # derivative of MSE loss
        dz2 = output * ( 1 - output )
        dL_dz2 = da2 * dz2
        dW2 = np.dot(self.a1.T, dL_dz2) / m
        db2 = np.sum(dL_dz2, axis=0, keepdims=True) / m

        # Hidden layer gradients
        da1 = np.dot(dL_dz2, self.W2.T)
        dz1 = da1 * relu_derivative(self.a1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        return dW1, db1, dW2, db2

    def update_weights(self, dW1, db1, dW2, db2):
        """Update weights using gradient descent"""
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    def train(self, X, epochs=100, batch_size=32):
        """Train the autoencoder"""
        losses = []
        n_samples = X.shape[0]

        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]

            epoch_loss = 0
            for i in range(0, n_samples, batch_size):
                batch = X_shuffled[i:i + batch_size]

                # Forward pass
                output = self.forward(batch)

                # Compute loss (MSE)
                loss = np.mean((output - batch) ** 2)
                epoch_loss += loss

                # Backward pass
                dW1, db1, dW2, db2 = self.backward(batch, output)

                # Update weights
                self.update_weights(dW1, db1, dW2, db2)

            avg_loss = epoch_loss / (n_samples // batch_size)
            losses.append(avg_loss)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

        return losses

    def encode(self, X):
        """Get latent representation"""
        z1 = np.dot(X, self.W1) + self.b1
        return relu(z1)

    def decode(self, latent):
        """Reconstruct from latent representation"""
        z2 = np.dot(latent, self.W2) + self.b2
        return sigmoid(z2)


# Load and preprocess data
print("Loading digits dataset...")
digits = load_digits()
X = digits.data / 16.0  # Normalize to [0, 1]
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
print(f"Input dimensions: {X_train.shape[1]}")

# Create and train autoencoder
input_size = X_train.shape[1]  # 64 (8x8 images)
hidden_size = 16  # Compress to 16 dimensions

print(f"\nCreating autoencoder: {input_size} -> {hidden_size} -> {input_size}")
autoencoder = AutoencoderScratch(input_size, hidden_size, learning_rate=0.1)

print("\nTraining...")
losses = autoencoder.train(X_train, epochs=50, batch_size=32)

# Test reconstruction
print("\nTesting reconstruction...")
test_sample = X_test[:10]
reconstructed = autoencoder.forward(test_sample)

# Compute test loss
test_loss = np.mean((reconstructed - test_sample) ** 2)
print(f"Test reconstruction loss: {test_loss:.6f}")

# Visualize latent space
print("\nEncoding test set to latent space...")
latent = autoencoder.encode(X_test)
print(f"Latent space shape: {latent.shape}")
print(f"Compression ratio: {input_size / hidden_size:.1f}x")

print("\n✓ From-scratch implementation complete!")
print(f"  - Learned to compress {input_size}D -> {hidden_size}D")
print(f"  - Final train loss: {losses[-1]:.6f}")
print(f"  - Test loss: {test_loss:.6f}")