# Variational Autoencoder (VAE) Implementation

This repository contains an implementation of a **Variational Autoencoder (VAE)** model, along with training, testing, and visualization scripts. Below is an overview of the files and instructions for setting up and running the code.

---

## Repository Contents

### Python Scripts:

1. **`vae.py`**
   - Contains the core implementation of the VAE model, including:
     - Encoder and Decoder networks
     - Reparameterization trick
     - Loss function implementation

2. **`train.py`**
   - Implements the training and testing procedures.
   - Includes dataset loading, model initialization, and training loop logic.
   - Saves models and logs training/testing losses for visualization.

3. **`plot.py`**
   - Script for plotting results such as the training and testing losses over epochs.

4. **Test Scripts**
   - Located in the `tests/` directory and contain unit tests for various components of the VAE model, including the encoder, decoder, and overall architecture.

---

## Environment Setup

This project requires Python and a virtual environment to manage dependencies. Follow the steps below to set up your environment using Conda:

1. **Create a Conda Virtual Environment from the venv.yml file**
   ```bash
   conda env create --name your-venv-name --file venv_dgm_prog2.yml
   ```

2. **Activate the Virtual Environment**
   ```bash
   conda activate your-venv-name
   ```

3. **Verify Setup**
   Ensure all dependencies are correctly installed by running:
   ```bash
   pytest
   ```

---

## Running the Code

### 1. Training the VAE Model

To train the VAE model, use the `train.py` script. Example usage:

```bash
python train.py --dataset mnist --epochs 50 --batch_size 128 --latent-dim 100 --lr 1e-3
```

**Key Arguments:**
- `--dataset`: Dataset to train on (e.g., `mnist`, `fashion-mnist`).
- `--epochs`: Number of training epochs.
- `--batch_size`: Batch size for training and testing.
- `--latent-dim`: Dimensionality of the latent space.
- `--lr`: Learning rate.

Results, including saved models and logs, will be stored in appropriate directories.

### 2. Visualizing Results

After training, use the `plot.py` script to visualize the loss curves:

```bash
python plot.py --losses_path .\logs\your-model-logs.pkl
```

---

## Testing the Code

### 1. Running All Tests
Run the unit tests for all components using `pytest`:
```bash
pytest
```

### 2. Running Specific Tests
To test a specific file, use the following format:
```bash
pytest tests\your-test.py
```
for example:
```bash
pytest tests\test_vae_functions.py
```

---

## Notes

- **Data Location**: Training and test data for MNIST and Fashion-MNIST will be automatically downloaded into a `data/` directory.
- **Samples**: Generated samples and saved models are stored in `samples/` and `models/` directories, respectively.
- **Logs**: Training and testing logs (e.g., reconstruction loss values) are saved in `logs/`.

---

## Additional Information

- Ensure a GPU is available for faster training. The code automatically detects and utilizes CUDA if available.
- Modify hyperparameters like the number of latent dimensions (`--latent-dim`) to experiment with the model.

---

For further inquiries or issues, feel free to contact the repository maintainer or create an issue.
