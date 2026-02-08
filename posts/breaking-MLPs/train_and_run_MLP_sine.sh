
# basic showcase of sine curve fitting
uv run python train.py --model_path model_tanh_1_10_10_10_1.pkl \
    --activation tanh \
    --layer_sizes 1 10 10 10 1 \
    --epochs 100000
    
uv run python visualize.py --model_path model_tanh_1_10_10_10_1.pkl
uv run python visualize.py --model_path model_tanh_1_10_10_10_1.pkl --suffix "wide" --period 10 --line_size 0.5

# how it works
uv run python train.py --model_path model_tanh_1_10_1.pkl \
    --activation tanh \
    --layer_sizes 1 10 1 \
    --epochs 200000 \
    --learning_rate 0.05
    
uv run python visualize.py --model_path model_tanh_1_10_1.pkl --animate


# what about a less complementary activation function?
uv run python train.py --model_path model_relu_1_10_10_10_1.pkl \
    --activation relu \
    --layer_sizes 1 10 10 10 1 \
    --epochs 200000
    
uv run python visualize.py --model_path model_relu_1_10_10_10_1.pkl

# more capacity
uv run python train.py --model_path model_relu_1_32_32_32_32_1.pkl \
    --activation relu \
    --layer_sizes 1 32 32 32 32 1 \
    --epochs 50000

uv run python visualize.py --model_path model_relu_1_32_32_32_32_1.pkl

# even more capacity
uv run python train.py --model_path model_relu_1_64_64_64_64_64_1.pkl \
    --activation relu \
    --layer_sizes 1 64 64 64 64 64 1 \
    --epochs 50000
    
uv run python visualize.py --model_path model_relu_1_64_64_64_64_64_1.pkl


# more data
uv run python train.py --model_path model_relu_1_64_64_64_64_64_1_more_data.pkl \
    --activation relu \
    --layer_sizes 1 64 64 64 64 64 1 \
    --epochs 20000 \
    --n_samples 10000
    
uv run python visualize.py --model_path model_relu_1_64_64_64_64_64_1_more_data.pkl


# what about extrapolation?

# see also tanh above

# maybe not that interesting
# uv run python visualize.py --model_path model_relu_1_10_10_10_1.pkl --suffix "wide" --period 10

uv run python visualize.py --model_path model_relu_1_64_64_64_64_64_1_more_data.pkl --suffix "wide" --period 10 --line_size 0.5


# bake in periodicity

uv run python train.py --model_path model_sin_1_10_1.pkl \
    --activation sin \
    --layer_sizes 1 10 1 \
    --epochs 20000 \
    --learning_rate 0.005

uv run python visualize.py --model_path model_sin_1_10_1.pkl --period 15 --line_size 0.5


uv run python train.py --model_path model_sin_1_15_15_15_1_square.pkl \
    --activation sin \
    --layer_sizes 1 15 15 15 1 \
    --epochs 200000 \
    --learning_rate 0.05 \
    --square True \
    --weight_decay 0.001

uv run python visualize.py --model_path model_sin_1_15_15_15_1_square.pkl
uv run python visualize.py --model_path model_sin_1_15_15_15_1_square.pkl --period 20 --suffix period_20 --line_size 0.5
uv run python visualize.py --model_path model_sin_1_15_15_15_1_square.pkl --period 201 --suffix period_201 --xlim -40 -45

# alterenative periodicty bake in


uv run python train.py --model_path model_fourier.pkl \
    --layer_sizes 1 15 15 15 1 \
    --activation relu \
    --square True \
    --fourier True \
    --learning_rate 0.01 \
    --epochs 100000 \
    --num_frequencies 10 \
    --min_freq 1 \
    --max_freq 5 \
    --weight_decay 0.001

uv run python visualize.py --model_path model_fourier.pkl
uv run python visualize.py --model_path model_fourier.pkl --period 20 --suffix period_20 --line_size 0.5
uv run python visualize.py --model_path model_fourier.pkl --period 200 --suffix period_200 --xlim -44 -47


# high frequency


uv run python train.py --model_path model_fourier_high_freq_15.pkl \
    --layer_sizes 1 15 15 15 1 \
    --activation relu \
    --square True \
    --high_frequency True \
    --fourier True \
    --learning_rate 0.005 \
    --epochs 250000 \
    --num_frequencies 10 \
    --min_freq 0.1 \
    --max_freq 5 \
    --weight_decay 0.001

uv run python visualize.py --model_path model_fourier_high_freq_15.pkl


uv run python train.py --model_path model_relu_extra20_high_freq_15.pkl \
    --layer_sizes 1 20 15 15 15 1 \
    --activation relu \
    --square True \
    --high_frequency True \
    --learning_rate 0.005 \
    --epochs 250000 \
    --weight_decay 0.001

uv run python visualize.py --model_path model_relu_extra20_high_freq_15.pkl

uv run python compare_loss.py \
    model_fourier_high_freq_15.pkl \
    model_relu_extra20_high_freq_15.pkl \
    --names "Fourier feature network" "ReLU only network" \
    --output loss_comparison_fourier_relu_high_freq.png