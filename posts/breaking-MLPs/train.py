# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "jax",
#     "jaxlib",
#     "matplotlib",
#     "numpy",
# ]
# ///

import jax
import jax.numpy as jnp
import numpy as np
import pickle
import argparse

def true_fn(x, high_frequency=False, square=False):
    """The underlying true function without noise."""
    if square:
        y = jnp.sign(np.sin(x))
    else:
        y = np.sin(x)
    if high_frequency:
        y += jnp.sign(np.sin(15 * x) / 3) * 0.2
    return y / 1.5

def generate_data(n_samples=1000, seed=42, high_frequency=False, period=3, square=False):
    """Generates noisy sine wave data."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-np.pi * period, np.pi * period, size=(n_samples, 1))
    noise_std = 0.01 if high_frequency else 0.1
    y = true_fn(x, high_frequency, square) + (noise_std * rng.standard_normal(size=(n_samples, 1)) / 1.2)
    return x / 10, y


def init_mlp_params(layer_sizes, key, activation="tanh"):
    """Initializes MLP parameters (weights and biases)."""
    params = []
    keys = jax.random.split(key, len(layer_sizes))
    
    # Iterate over layers to create weights and biases
    for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        k1, k2 = jax.random.split(keys[i])
        
        if activation in ["relu", "relu6"]:
            # He initialization: sqrt(2/n_in) for ReLU and variants
            w = jax.random.normal(k1, (n_in, n_out)) * jnp.sqrt(2.0 / n_in)
            # Small positive bias to help with "dead" neurons
            b = jnp.ones((n_out,)) * 0.01
        elif activation == "sin":
            # SIREN initialization (assuming omega_0 = 30 included in weights)
            omega_0 = 120.0
            if i == 0:
                # First layer: uniform(-omega_0/n_in, omega_0/n_in)
                bound = omega_0 / n_in
            else:
                # Hidden layers: uniform(-sqrt(6/n_in), sqrt(6/n_in))
                bound = jnp.sqrt(6.0 / n_in)
            
            w = jax.random.uniform(k1, (n_in, n_out), minval=-bound, maxval=bound)
            b = jnp.zeros((n_out,))
        else:
            # Xavier/Glorot initialization: sqrt(1/n_in) for tanh/sigmoid
            w = jax.random.normal(k1, (n_in, n_out)) * jnp.sqrt(1.0 / n_in)
            b = jax.random.normal(k2, (n_out,)) * 0.1
            
        params.append({'w': w, 'b': b})
    return params


def forward(params, x, activation="tanh", fourier=False, num_frequencies=6, max_freq=20.0, min_freq=1.0):
    """Forward pass of the MLP."""
    # Inputs are (batch_size, input_dim)
    activations = fourier_encode(x, num_frequencies=num_frequencies, max_freq=max_freq, min_freq=min_freq) if fourier else x
    
    # Loop through hidden layers
    for layer in params[:-1]:
        activations = jnp.dot(activations, layer['w']) + layer['b']
        if activation == "tanh":
            activations = jax.nn.tanh(activations)
        elif activation == "relu":
            activations = jax.nn.leaky_relu(activations, negative_slope=0.05)
        elif activation == "relu6":
            activations = jax.nn.relu6(activations)
        elif activation == "sigmoid":
            activations = jax.nn.sigmoid(activations)
        elif activation == "sin":
            activations = jnp.sin(activations)
        elif activation == "linear":
            pass
        else:
            raise ValueError(f"Unknown activation function: {activation}")
        
    # Output layer (no activation for regression)
    final_layer = params[-1]
    logits = jnp.dot(activations, final_layer['w']) + final_layer['b']
    return logits

def fourier_encode(x, num_frequencies=12, max_freq=20.0, min_freq=1.0):
    """
    Cleaner Fourier Encoding: sin(n * (x * 10))
    This aligns 'n' with the actual harmonics of your sin(1.0 * x) signal.
    """
    # Create the frequency matrix B (harmonics: 1, 2, 3...)
    # Since model gets xScaled = xRaw/10, we multiply by 10 to get back to radians.
    freqs = jnp.linspace(min_freq, max_freq, num_frequencies)
    B = (freqs * 10.0).reshape(1, -1)
    print(B)
    
    x_proj = jnp.dot(x, B)
    
    # Concatenate [sin, cos]
    return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)

def get_activations(params, x, activation="tanh", fourier=False, num_frequencies=6, max_freq=20.0, min_freq=1.0):
    """Returns the activations of all hidden layers."""
    all_activations = []
    activations = fourier_encode(x, num_frequencies=num_frequencies, max_freq=max_freq, min_freq=min_freq) if fourier else x
    for layer in params[:-1]:
        activations = jnp.dot(activations, layer['w']) + layer['b']
        if activation == "tanh":
            activations = jax.nn.tanh(activations)
        elif activation == "relu":
            activations = jax.nn.leaky_relu(activations, negative_slope=0.05)
        elif activation == "relu6":
            activations = jax.nn.relu6(activations)
        elif activation == "sigmoid":
            activations = jax.nn.sigmoid(activations)
        elif activation == "sin":
            activations = jnp.sin(activations)
        elif activation == "linear":
            pass
        else:
            raise ValueError(f"Unknown activation function: {activation}")
        all_activations.append(activations)
    return all_activations


def loss_fn(params, x, y, activation="tanh", fourier=False, num_frequencies=6, max_freq=20.0, min_freq=1.0):
    """Mean Squared Error Loss."""
    predictions = forward(params, x, activation, fourier, num_frequencies, max_freq, min_freq)
    return jnp.mean((predictions - y) ** 2)


from functools import partial

@partial(jax.jit, static_argnums=(3, 4, 5, 6, 7, 8))
def update(params, x, y, learning_rate=0.01, activation="tanh", fourier=False, num_frequencies=6, max_freq=20.0, min_freq=1.0, weight_decay=1e-4):
    """Performs one gradient descent update step with Weight Decay (L2)."""
    
    def loss_with_reg(p, x_val, y_val, act, four, n_freq, m_freq, min_f):
        mse_loss = loss_fn(p, x_val, y_val, act, four, n_freq, m_freq, min_f)
        # L2 Regularization (Weight Decay)
        l2_reg = 0.5 * sum(jnp.sum(jnp.square(layer['w'])) for layer in p)
        return mse_loss + weight_decay * l2_reg

    # jax.value_and_grad returns both the loss value and the gradients
    loss, grads = jax.value_and_grad(loss_with_reg)(params, x, y, activation, fourier, num_frequencies, max_freq, min_freq)
    
    def update_layer(parameters, gradient):
        return parameters - learning_rate * gradient
    
    # Apply update to all layers using jax.tree.map
    new_params = jax.tree.map(update_layer, params, grads)
    
    return loss, new_params


def train_model(args):
    print("Initializing JAX Sine Wave Training...")
    
    # 1. Generate Data
    key = jax.random.PRNGKey(args.seed)
    x_train, y_train = generate_data(n_samples=args.n_samples, high_frequency=args.high_frequency, square=args.square)
    
    # 2. Initialize Model
    layer_sizes = list(args.layer_sizes)
    if args.fourier:
        # Input dimension becomes 2 * num_frequencies
        layer_sizes[0] = 2 * args.num_frequencies
    params = init_mlp_params(layer_sizes, key, activation=args.activation)
    print(f"Model initialized with layer sizes: {layer_sizes}")

    # 3. Training Loop
    losses = []
    epochs = args.epochs
    learning_rate = args.learning_rate
    
    print(f"Starting training for {epochs} epochs (Weight Decay: {args.weight_decay})...")
    for epoch in range(epochs):
        loss, params = update(params, x_train, y_train, learning_rate, args.activation, args.fourier, args.num_frequencies, args.max_freq, args.min_freq, weight_decay=args.weight_decay)
        losses.append(loss)
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch:05d} | Loss: {loss:.6f}")
            
    print(f"Final Loss: {loss:.6f}")
    
    # Save model
    print(f"Saving model to {args.model_path}...")
    with open(args.model_path, 'wb') as f:
        pickle.dump({
            'params': params, 
            'losses': losses, 
            'layer_sizes': layer_sizes,
            'fourier': args.fourier,
            'num_frequencies': args.num_frequencies,
            'max_freq': args.max_freq,
            'min_freq': args.min_freq,
            'activation': args.activation,
            'square': args.square,
            'high_frequency': args.high_frequency,
            'n_samples': args.n_samples
        }, f)
    print("Model saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='JAX Sine Wave Training')
    parser.add_argument('--model_path', type=str, default='model.pkl', help='Path to save the model')
    parser.add_argument('--activation', type=str, default='tanh', help='Activation function to use')
    parser.add_argument('--layer_sizes', type=int, nargs='+', default=[1, 32, 32, 32, 1], help='Layer sizes')
    parser.add_argument('--epochs', type=int, default=50000, help='Number of epochs to train for')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--n_samples', type=int, default=1000, help='Number of training samples')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--high_frequency', type=bool, default=False, help='Use high frequency data')
    parser.add_argument('--square', type=bool, default=False, help='Use square wave data')
    parser.add_argument('--fourier', type=bool, default=False, help='Use Fourier encoding')
    parser.add_argument('--num_frequencies', type=int, default=12, help='Number of Fourier frequencies')
    parser.add_argument('--min_freq', type=float, default=1.0, help='Min Fourier frequency')
    parser.add_argument('--max_freq', type=float, default=20.0, help='Max Fourier frequency')
    parser.add_argument('--weight_decay', type=float, default=0, help='L2 regularization strength')
    
    args = parser.parse_args()
    train_model(args)
