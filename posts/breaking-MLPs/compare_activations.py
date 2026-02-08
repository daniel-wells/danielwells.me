import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import plotnine as pn

# Define the activation functions using jax.numpy for autodiff
def relu(x):
    return jnp.maximum(0, x)

def leaky_relu(x, alpha=0.1):
    return jnp.where(x > 0, x, alpha * x)

def tanh(x):
    return jnp.tanh(x)

def gelu_approx(x):
    """Tanh approximation of GeLU."""
    return 0.5 * x * (1 + jnp.tanh(jnp.sqrt(2 / jnp.pi) * (x + 0.044715 * x**3)))

def gelu_exact(x):
    """Full non-approximate GeLU using the error function."""
    return 0.5 * x * (1 + jax.scipy.special.erf(x / jnp.sqrt(2)))

def softplus(x, beta=1):
    """Softplus activation function with beta parameter."""
    # Using a numerically stable implementation
    return (1.0 / beta) * jnp.log1p(jnp.exp(-jnp.abs(beta * x))) + jnp.maximum(0, x)

def mish(x):
    """Mish activation function: x * tanh(softplus(x))."""
    return x * jnp.tanh(softplus(x, beta=1))

def swish(x):
    """Swish (SiLU) activation function: x * sigmoid(x)."""
    return x / (1 + jnp.exp(-x))

# Functions for which we want to compute derivatives automatically
# Some need partial application for fixed parameters
funcs = {
    'ReLU': relu,
    'Leaky ReLU': leaky_relu,
    'Tanh': tanh,
    'GeLU (Approx)': gelu_approx,
    'GeLU (Exact)': gelu_exact,
    'Softplus (β=4)': lambda x: softplus(x, beta=4),
    'Mish': mish,
    'Swish': swish
}

# Generate x values
x_vals = jnp.linspace(-3, 3, 600)

results = []
for name, func in funcs.items():
    # Compute value
    y_vals = jax.vmap(func)(x_vals)
    # Compute first derivative using jax.grad
    dy_vals = jax.vmap(jax.grad(func))(x_vals)
    
    for x, y, dy in zip(x_vals, y_vals, dy_vals):
        results.append({'x': float(x), 'Value': float(y), 'Derivative': float(dy), 'Activation': name})

df = pd.DataFrame(results)

# Reorder the Activation factor to bring ReLU and GeLU to the forefront (drawn last)
activation_order = [
    'Tanh', 'Softplus (β=4)', 'Leaky ReLU', 'Mish', 'Swish', 
    'GeLU (Approx)', 'GeLU (Exact)', 'ReLU'
]
df['Activation'] = pd.Categorical(df['Activation'], categories=activation_order, ordered=True)

# Shared aesthetics
common_theme = pn.theme_minimal() + pn.theme(
    figure_size=(10, 6),
    legend_position='right',
    plot_title=pn.element_text(size=14, weight='bold'),
    plot_subtitle=pn.element_text(size=10, style='italic')
)

color_scale = pn.scale_color_manual(values={
    'ReLU': '#e41a1c',
    'Leaky ReLU': '#377eb8',
    'Tanh': '#4daf4a',
    'GeLU (Approx)': '#984ea3',
    'GeLU (Exact)': '#ff7f00',
    'Softplus (β=4)': '#525252',
    'Mish': '#a65628',
    'Swish': '#f781bf'
})

linetype_scale = pn.scale_linetype_manual(values={
    'ReLU': 'solid',
    'Leaky ReLU': 'solid',
    'Tanh': 'solid',
    'GeLU (Approx)': 'solid',
    'GeLU (Exact)': 'dotted',
    'Softplus (β=4)': 'solid',
    'Mish': 'solid',
    'Swish': 'solid'
}, guide=None)

# 1. Plot Activations
plot_vals = (
    pn.ggplot(df, pn.aes(x='x', y='Value', color='Activation', linetype='Activation')) +
    pn.geom_line(size=1.2) +
    pn.labs(x='Input (x)', y='Output (f(x))', color='Function') +
    common_theme + color_scale + linetype_scale
)
plot_vals.save('activation_comparison.png', dpi=150)

# 2. Plot Derivatives
plot_grads = (
    pn.ggplot(df, pn.aes(x='x', y='Derivative', color='Activation', linetype='Activation')) +
    pn.geom_line(size=1.2) +
    pn.labs(x='Input (x)', y="Derivative (f'(x))", color='Function') +
    common_theme + color_scale + linetype_scale
)
plot_grads.save('activation_derivative_comparison.png', dpi=150)
