import pickle
import argparse
import pandas as pd
import plotnine as pn
import os

def compare_losses(model_paths, names=None, output_path="loss_comparison.png"):
    data = []
    
    if names and len(names) != len(model_paths):
        print(f"Error: Number of names ({len(names)}) does not match number of models ({len(model_paths)})")
        return

    for i, path in enumerate(model_paths):
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping...")
            continue
            
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
            losses = model_data.get('losses', [])
            
            if names:
                model_display_name = names[i]
            else:
                model_display_name = os.path.basename(path).replace('.pkl', '')
            
            # Create entries for each epoch
            for epoch, loss in enumerate(losses):
                # Sample every 10 epochs if training was long
                if epoch % 10 == 0:
                    data.append({
                        'Epoch': epoch,
                        'Loss': float(loss),
                        'Model': model_display_name
                    })
    
    if not data:
        print("No data found to plot.")
        return

    df = pd.DataFrame(data)
    
    plot = (
        pn.ggplot(df, pn.aes(x='Epoch', y='Loss', color='Model')) +
        pn.geom_line(size=0.8, alpha=0.8) +
        pn.scale_y_log10() +
        pn.labs(
            title='Training Loss Comparison',
            x='Epoch',
            y='MSE Loss (Log Scale)',
            color='Model'
        ) +
        pn.theme_minimal() +
        pn.scale_color_brewer(type='qual', palette='Set1') +
        pn.theme(legend_position='bottom')
    )
    
    plot.save(output_path, width=10, height=6, dpi=150)
    print(f"Comparison plot saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare training losses of multiple JAX models')
    parser.add_argument('model_paths', type=str, nargs='+', help='Paths to .pkl model files')
    parser.add_argument('--names', type=str, nargs='+', help='Custom names for the models in the plot legend')
    parser.add_argument('--output', type=str, default='loss_comparison.png', help='Output filename')
    
    args = parser.parse_args()
    compare_losses(args.model_paths, args.names, args.output)
