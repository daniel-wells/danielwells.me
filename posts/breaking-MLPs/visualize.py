# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "jax",
#     "jaxlib",
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "plotnine",
# ]
# ///

import numpy as np
import pickle
import argparse
import pandas as pd
import plotnine as pn
import train # Import model definition and helpers
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

def plot_prediction(x_train, y_train, x_test, y_pred, output_dir, high_frequency=False, square=False, line_size=0.75, xlim=None):
    """Generates and saves the prediction vs ground truth plot."""
    # JAX arrays to numpy and flattening
    x_train_np = np.array(x_train).flatten()
    y_train_np = np.array(y_train).flatten()
    x_test_np = np.array(x_test).flatten()
    y_pred_np = np.array(y_pred).flatten()
    y_true = train.true_fn(x_test_np * 10, high_frequency=high_frequency, square=square)
    
    label_name = 'Square Wave' if square else ('High Freq Sine' if high_frequency else 'Sine Wave')
    df_train = pd.DataFrame({'x': x_train_np, 'y': y_train_np, 'dataset': 'Training Data'})
    df_test = pd.DataFrame({'x': x_test_np, 'y': y_true.flatten(), 'dataset': f'True {label_name}'})
    df_pred = pd.DataFrame({'x': x_test_np, 'y': y_pred_np, 'dataset': 'Model Prediction'})
    
    plot = (
        pn.ggplot() +
        # Training data as points
        pn.geom_point(pn.aes(x='x', y='y', color='dataset'), data=df_train, alpha=0.4, size=0.5) +
        # True function and Prediction as lines
        pn.geom_line(pn.aes(x='x', y='y', color='dataset'), data=df_test, size=line_size, linetype='dashed') +
        pn.geom_line(pn.aes(x='x', y='y', color='dataset'), data=df_pred, size=line_size*1.5) +
        pn.scale_color_manual(values={
            'Training Data': 'blue',
            f'True {label_name}': 'green',
            'Model Prediction': 'red'
        }) +
        pn.labs(
            title=f'Prediction vs Ground Truth ({label_name})', 
            x='x', y='y', color=""
        ) +
        pn.theme_minimal() +
        pn.theme(legend_position='bottom')
    )
    if xlim:
        plot += pn.xlim(xlim[0], xlim[1])
        
    filename = os.path.join(output_dir, 'plot_prediction.png')
    plot.save(filename, width=9, height=6, dpi=100)
    print(f"Saved {filename}")

def plot_loss(losses, output_dir):
    """Generates and saves the training loss plot."""
    df_loss = pd.DataFrame({'epoch': np.arange(len(losses)), 'loss': np.array(losses).flatten()})
    
    plot = (
        pn.ggplot(df_loss, pn.aes(x='epoch', y='loss')) +
        pn.geom_line(color='blue') +
        pn.scale_y_log10() +
        pn.labs(title='Training Loss Over Time', x='Epoch', y='MSE Loss') +
        pn.theme_minimal()
    )
    filename = os.path.join(output_dir, 'plot_loss.png')
    plot.save(filename, width=9, height=6, dpi=100)
    print(f"Saved {filename}")

def plot_activations(params, x_test, output_dir, activation="tanh", high_frequency=False, square=False, fourier=False, num_frequencies=6, max_freq=20.0, min_freq=1.0):
    """Generates and saves weighted activations plot for each hidden layer."""
    all_activations = train.get_activations(params, x_test, activation=activation, fourier=fourier, num_frequencies=num_frequencies, max_freq=max_freq, min_freq=min_freq)
    x_test_np = np.array(x_test).flatten()
    
    for layer_idx, activations in enumerate(all_activations):
        # Weight by the next layer's weights
        next_w = params[layer_idx + 1]['w'] if layer_idx + 1 < len(params) else params[-1]['w']
        
        is_last_hidden = (layer_idx == len(all_activations) - 1)
        
        if is_last_hidden:
            weighted_activations = activations * next_w.T
        else:
            importance = np.sum(np.abs(next_w), axis=1)
            weighted_activations = activations * importance
        
        weighted_activations_np = np.array(weighted_activations)
        n_samples, n_neurons = weighted_activations_np.shape
        
        df_activations = pd.DataFrame(weighted_activations_np, columns=[f"neuron_{i}" for i in range(n_neurons)])
        df_activations['x'] = x_test_np
        
        sum_vals = weighted_activations_np.sum(axis=1)
        if is_last_hidden:
            bias_val = np.array(params[-1]['b']).item()
            sum_vals += bias_val
            df_bias = pd.DataFrame({'x': x_test_np, 'y': bias_val, 'label': 'Final Bias'})
            
        df_sum = pd.DataFrame({'x': x_test_np, 'y': sum_vals, 'label': 'Sum (Reconstruction)'})
        y_true = train.true_fn(x_test_np * 10, high_frequency=high_frequency, square=square)
        label_name = 'Square Wave' if square else 'Sine Wave'
        df_true = pd.DataFrame({'x': x_test_np, 'y': y_true.flatten(), 'label': f'True {label_name}'})
        
        df_activations_long = df_activations.melt(id_vars=['x'], var_name='neuron', value_name='activation')
        
        plot = (
            pn.ggplot() +
            # Individual weighted activations (mapped to color for legend)
            pn.geom_line(pn.aes(x='x', y='activation', group='neuron', color='"Individual neuron output"'), 
                         data=df_activations_long, alpha=0.5, size=0.4) +
            pn.scale_color_manual(values={
                'Individual neuron output': 'purple',
                'Sum (Reconstruction)': 'black', 
                f'True {label_name}': 'green',
                'Final Bias': 'orange'
            }) +
            pn.labs(
                title=f'Layer {layer_idx + 1} Weighted Activations',
                subtitle='Individual neuron contributions scaled by importance' if not is_last_hidden else 'Individual contributions summed to reconstruct the signal',
                x='x', y='Value', color=""
            ) +
            pn.theme_minimal() + 
            pn.theme(legend_position='bottom')
        )
        
        if is_last_hidden:
            # The sum
            plot += pn.geom_line(pn.aes(x='x', y='y', color='label'), data=df_sum, size=1.5)
            plot += pn.geom_line(pn.aes(x='x', y='y', color='label'), data=df_bias, linetype='dotted', size=1.0)
            plot += pn.geom_line(pn.aes(x='x', y='y', color='label'), data=df_true, linetype='dashed', size=1.0)
            plot += pn.labs(title=f'Layer {layer_idx + 1} (Last Hidden): Reconstructing the {label_name}')

        filename = os.path.join(output_dir, f'plot_activations_layer{layer_idx + 1}.png')
        plot.save(filename, width=9, height=6, dpi=100)
        print(f"Saved {filename}")


def plot_stacked_activations(params, x_test, output_dir, n_bars=30, activation="tanh", fourier=False, num_frequencies=6, max_freq=20.0, min_freq=1.0):
    """Generates a shifted waterfall-style stacked bar chart of contributions."""
    print("Generating shifted waterfall activation plot...")
    all_activations = train.get_activations(params, x_test, activation=activation, fourier=fourier, num_frequencies=num_frequencies, max_freq=max_freq, min_freq=min_freq)
    last_activations = all_activations[-1]
    last_w = params[-1]['w']
    last_b = params[-1]['b']
    
    # Calculate weighted contributions
    weighted_contribs = np.array(last_activations * last_w.T) # (n_samples, n_neurons)
    
    # Downsample
    n_samples = len(x_test)
    indices = np.linspace(0, n_samples - 1, n_bars, dtype=int)
    x_sub = np.array(x_test[indices]).flatten()
    contribs_sub = weighted_contribs[indices]
    bias_sub = np.full((n_bars, 1), np.array(last_b))
    
    data_all = np.hstack([contribs_sub, bias_sub])
    
    rects = []
    # Width of each bar
    bar_width = (x_sub[1] - x_sub[0]) * 0.25
    dx = bar_width * 0.7
    
    for i in range(n_bars):
        vals = data_all[i, :]
        x = x_sub[i]
        
        # 1. Stack positive components UP from 0 (Left Side)
        pos_indices = np.where(vals >= 0)[0]
        curr_y = 0
        x_pos = x - dx
        for idx in pos_indices:
            val = vals[idx]
            rects.append({
                'xmin': x_pos - bar_width/2, 'xmax': x_pos + bar_width/2,
                'ymin': curr_y, 'ymax': curr_y + val,
                'component': f"comp_{idx}",
                'border_color': 'green'
            })
            curr_y += val
        
        total_pos = curr_y
        
        # 2. Stack negative components DOWN from total_pos (Right Side)
        neg_indices = np.where(vals < 0)[0]
        curr_y = total_pos
        x_neg = x + dx
        for idx in neg_indices:
            val = vals[idx] # val is negative
            rects.append({
                'xmin': x_neg - bar_width/2, 'xmax': x_neg + bar_width/2,
                'ymin': curr_y + val, 'ymax': curr_y,
                'component': f"comp_{idx}",
                'border_color': 'red'
            })
            curr_y += val
            
    df_rects = pd.DataFrame(rects)
    
    # Net sum for line overlay (actual prediction)
    df_sum = pd.DataFrame({
        'x': x_sub,
        'y': data_all.sum(axis=1)
    })
    
    plot = (
        pn.ggplot() +
        # Waterfall blocks
        pn.geom_rect(
            pn.aes(xmin='xmin', xmax='xmax', ymin='ymin', ymax='ymax', fill='component', color='border_color'),
            data=df_rects,
            alpha=0.8,
            size=1.85
        ) +
        # Net sum (the actual sine wave) - centered at x
        pn.geom_line(
            pn.aes(x='x', y='y'),
            data=df_sum,
            color='black',
            size=1.2,
            alpha=0.75
        ) +
        pn.scale_fill_brewer(type='qual', palette='Set3') +
        pn.scale_color_identity() +
        pn.labs(
            title='Last Layer: Shifted Waterfall Chart',
            subtitle='Green border: Positive contributions | Red border: Negative contributions\n Black line: Net Sum (Prediction)',
            x='x', y='Value', fill='Component'
        ) +
        pn.theme_minimal() +
        pn.theme(legend_position='none')
    )
    
    filename = os.path.join(output_dir, 'plot_activations_waterfall_shifted.png')
    plot.save(filename, width=11, height=6, dpi=100)
    print(f"Saved {filename}")

def animate_reconstruction(params, x_test, output_dir, activation="tanh", high_frequency=False, square=False, fourier=False, num_frequencies=6, max_freq=20.0, min_freq=1.0):
    """Creates an animation showing the cumulative sum of neuron contributions."""
    print("Generating reconstruction animation...")
    all_activations = train.get_activations(params, x_test, activation=activation, fourier=fourier, num_frequencies=num_frequencies, max_freq=max_freq, min_freq=min_freq)
    last_activations = np.array(all_activations[-1])
    last_w = np.array(params[-1]['w'])
    last_b = np.array(params[-1]['b'])
    x_np = np.array(x_test).flatten()
    
    contribs = last_activations * last_w.T
    
    importance = np.var(contribs, axis=0)
    sorted_indices = np.argsort(importance)[::-1]
    sorted_contribs = contribs[:, sorted_indices]
    
    n_neurons = sorted_contribs.shape[1]
    
    # Calculate dynamic Y limits
    cum_sums = sorted_contribs.cumsum(axis=1) + last_b
    y_min_val = min(sorted_contribs.min(), cum_sums.min(), -1.2)
    y_max_val = max(sorted_contribs.max(), cum_sums.max(), 1.2)
    margin = (y_max_val - y_min_val) * 0.1
    
    fig, ax = plt.subplots(figsize=(8, 5), dpi=80)
    ax.set_xlim(x_np.min(), x_np.max())
    ax.set_ylim(y_min_val - margin, y_max_val + margin)
    ax.set_title("Iterative Sine Wave Reconstruction")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
    # Ground truth
    y_true = train.true_fn(x_np * 10, high_frequency=high_frequency, square=square)
    label_name = 'Square Wave' if square else 'Sine Wave'
    ax.plot(x_np, y_true.flatten(), 'g--', alpha=0.3, label=f'True {label_name}')
    
    line_component, = ax.plot([], [], 'purple', alpha=0.6, label='Curve to be added', lw=1.5)
    line_cumulative, = ax.plot([], [], 'black', lw=2, label='Cumulative Sum (+ bias)')
    text_info = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontweight='bold')
    ax.legend(loc='lower right')

    def init():
        line_component.set_data([], [])
        line_cumulative.set_data([], [])
        text_info.set_text('')
        return line_component, line_cumulative, text_info

    def update(frame_data):
        neuron_idx, phase = frame_data
        comp = sorted_contribs[:, neuron_idx]
        line_component.set_data(x_np, comp)
        
        if phase == 0:
            if neuron_idx == 0:
                cum_sum = np.full_like(x_np, last_b)
            else:
                cum_sum = sorted_contribs[:, :neuron_idx].sum(axis=1) + last_b
            text_info.set_text(f'Neuron {neuron_idx+1}/{n_neurons}')
        else:
            cum_sum = sorted_contribs[:, :neuron_idx+1].sum(axis=1) + last_b
            text_info.set_text(f'Neuron {neuron_idx+1}/{n_neurons}')
            
        line_cumulative.set_data(x_np, cum_sum)
        return line_component, line_cumulative, text_info

    frame_sequence = []
    for i in range(n_neurons):
        p0_duration = max(1, int(8 * (0.8**i))) if i < 15 else 1
        p1_duration = max(1, int(4 * (0.8**i))) if i < 15 else 1
        frame_sequence.extend([(i, 0)] * p0_duration)
        frame_sequence.extend([(i, 1)] * p1_duration)

    anim = FuncAnimation(fig, update, frames=frame_sequence, init_func=init, blit=True)
    
    filename = os.path.join(output_dir, 'reconstruction_animation.gif')
    try:
        anim.save(filename, writer='pillow', fps=8, dpi=300)
        print(f"Saved {filename}")
    except Exception as e:
        print(f"Could not save animation: {e}")
    
    plt.close()

def analyze_model(args):
    # Determine output directory
    model_name = os.path.basename(args.model_path)
    output_dir = model_name.replace('.pkl', '')
    if args.suffix:
        output_dir += f"_{args.suffix}"
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    print(f"Loading model from {args.model_path}...")
    try:
        with open(args.model_path, 'rb') as f:
            data = pickle.load(f)
            params = data['params']
            losses = data.get('losses', [])
            layer_sizes = data.get('layer_sizes', [])
            fourier = data.get('fourier', False)
            num_frequencies = data.get('num_frequencies', 6)
            max_freq = data.get('max_freq', 20.0)
            activation = data.get('activation', 'tanh')
            square = data.get('square', False)
            high_frequency = data.get('high_frequency', False)
            n_samples = data.get('n_samples', 1000)
            min_freq = data.get('min_freq', 1.0)
    except FileNotFoundError:
        print(f"Error: Model file '{args.model_path}' not found. Run training first.")
        return

    print(f"Generating plots for '{model_name}' inside '{output_dir}/'...")
    
    # Generate data (Representing the original training distribution)
    x_train, y_train = train.generate_data(n_samples=n_samples, high_frequency=high_frequency, square=square)
    x_test = np.linspace(-np.pi * args.period, np.pi * args.period, 1000 * int(args.period)).reshape(-1, 1) / 10
    y_pred = train.forward(params, x_test, activation=activation, fourier=fourier, num_frequencies=num_frequencies, max_freq=max_freq, min_freq=min_freq)
    
    # Call all plotting functions with output_dir
    plot_prediction(x_train, y_train, x_test, y_pred, output_dir, high_frequency=high_frequency, square=square, line_size=args.line_size, xlim=args.xlim)
    plot_loss(losses, output_dir)
    plot_activations(params, x_test, output_dir, activation=activation, high_frequency=high_frequency, square=square, fourier=fourier, num_frequencies=num_frequencies, max_freq=max_freq, min_freq=min_freq)
    plot_stacked_activations(params, x_test, output_dir, activation=activation, fourier=fourier, num_frequencies=num_frequencies, max_freq=max_freq, min_freq=min_freq)
    if args.animate:
        animate_reconstruction(params, x_test, output_dir, activation=activation, high_frequency=high_frequency, square=square, fourier=fourier, num_frequencies=num_frequencies, max_freq=max_freq, min_freq=min_freq)
    plot_network(params, layer_sizes, output_dir)

def plot_network(params, layer_sizes, output_dir):
    """Visualizes the neural network weights with greedy sorting."""
    print("Generating network visualization (sorted)...")
    
    permutations = [np.arange(size) for size in layer_sizes]
    nodes = []
    node_pos_map = {}
    
    for i, size in enumerate(layer_sizes):
        perm = permutations[i]
        for rank, orig_idx in enumerate(perm):
            y = (size / 2 - 0.5) - rank
            # Input layer has no bias, others do
            bias = 0.0
            if i > 0:
                bias = float(params[i-1]['b'][orig_idx])
            
            nodes.append({
                'layer': i, 
                'node_idx': orig_idx, 
                'x': i, 
                'y': y, 
                'bias': bias,
                'abs_bias': abs(bias)
            })
            node_pos_map[(i, orig_idx)] = {'x': i, 'y': y}
            
    df_nodes = pd.DataFrame(nodes)
    
    edges = []
    for i, param in enumerate(params):
        w = np.array(param['w'])
        n_in, n_out = w.shape
        for r in range(n_in):
            for c in range(n_out):
                weight = w[r, c]
                src_pos = node_pos_map[(i, r)]
                tgt_pos = node_pos_map[(i + 1, c)]
                edges.append({
                    'x': src_pos['x'], 'y': src_pos['y'],
                    'xend': tgt_pos['x'], 'yend': tgt_pos['y'],
                    'weight': weight,
                    'abs_weight': abs(weight)
                })
                
    df_edges = pd.DataFrame(edges)
    df_edges = df_edges.sort_values('abs_weight', ascending=True)
    
    plot = (
        pn.ggplot() +
        pn.geom_segment(
            pn.aes(x='x', y='y', xend='xend', yend='yend', alpha='abs_weight', size='abs_weight', color='weight'),
            data=df_edges
        ) +
        pn.scale_color_gradient2(low='red', mid='black', high='blue', midpoint=0, name='Weight') + 
        pn.geom_point(
            pn.aes(x='x', y='y', fill='bias'),
            data=df_nodes,
            size=2.5,
            color='black',
            stroke=0.5
        ) +
        pn.scale_fill_gradient2(low='red', mid='white', high='blue', midpoint=0, name='Bias') + 
        pn.labs(x='Layer', y='Neuron Index') +
        pn.theme_void() + 
        pn.theme(
            plot_background=pn.element_rect(fill='white', color='white'),
            legend_position='bottom'
        ) +
        pn.scale_size_continuous(range=(0.01, 1.5), guide=None) + 
        pn.scale_alpha_continuous(range=(0.1, 1), guide=None)
    )
    
    filename = os.path.join(output_dir, 'plot_network.png')
    plot.save(filename, width=9, height=6, dpi=100)
    print(f"Saved {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='JAX Sine Wave Visualization')
    parser.add_argument('--model_path', type=str, default='model.pkl', help='Path to load the model from')
    parser.add_argument('--period', type=float, default=3, help='Repeats of the sine wave to visualize (can be larger than training period)')
    parser.add_argument('--animate', action='store_true', help='Generate reconstruction animation (can be slow)')
    parser.add_argument('--suffix', type=str, default='', help='Optional suffix for the output directory')
    parser.add_argument('--line_size', type=float, default=0.75, help='Line size for the prediction plot')
    parser.add_argument('--xlim', type=float, nargs=2, default=None, help='X-axis limits for the prediction plot (min max)')

    args = parser.parse_args()
    analyze_model(args)
