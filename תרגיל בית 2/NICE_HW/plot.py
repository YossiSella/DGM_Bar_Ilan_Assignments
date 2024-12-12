import os
import pickle
import matplotlib.pyplot as plt

def plot_losses(losses_path):
    """Load losses from file and plot them."""
    # Load losses from the pickle file
    with open(losses_path, 'rb') as f:
        losses = pickle.load(f)

    # Check that both train and test losses are present
    if 'train_nll' not in losses or 'test_nll' not in losses:
        raise KeyError("The losses file must contain 'train_nll' and 'test_nll' keys.")
    
    # Extract train and test losses
    train_losses = losses['train_nll']
    test_losses  = losses['test_nll']
    epochs = range(1, len(train_losses) + 1)

    # Additional check to ensure they're not empty
    if not train_losses or not test_losses:
        raise ValueError("Train or test losses are empty. Please check the input file.")

    print(f"Train Losses: {len(train_losses)} epochs loaded.")
    print(f"Test Losses: {len(test_losses)} epochs loaded.")

      # Extract the pickle file name (without extension)
    pickle_name = os.path.basename(losses_path)
    pickle_name_without_ext = os.path.splitext(pickle_name)[0]

    # Plot losses
    # Create a figure and primary y-axis
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Set the window title
    fig.canvas.manager.set_window_title(pickle_name_without_ext)

    # Plot train losses on the primary y-axis
    ax1.plot(epochs, train_losses, label="Train Loss", color="blue")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Train Loss", color="blue")
    ax1.tick_params(axis='y', labelcolor="blue")

    # Create a secondary y-axis for test losses
    ax2 = ax1.twinx()
    ax2.plot(epochs, test_losses, label="Test Loss", color="red")
    ax2.set_ylabel("Test Loss", color="red")
    ax2.tick_params(axis='y', labelcolor="red")

    # Add a title and legend
    plt.title("Train and Test Loss Over Epochs")
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot losses from a saved pickle file")
    parser.add_argument('--losses_path', type=str, required=True, help='Path to the pickle file containing losses')

    args = parser.parse_args()
    plot_losses(args.losses_path)