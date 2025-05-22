

import argparse
import torch, torchvision
from torchvision import transforms, utils
import numpy as np
from DDPM import DDPM
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os
import pickle


def train(model, trainloader, optimizer, epoch,device):

    progress_bar = tqdm(trainloader, total=len(trainloader), desc=f"Training Epoch {epoch:02d}")
    losses = []

    model.train()  # set to training mode
    for image, target in progress_bar:
        noise  = torch.randn_like(image).to(device)
        image  = image.to(device)
        target = target.to(device)
        
        t = torch.randint(low=0, high=model.timesteps, size=(image.size(0),), device=device).long()

        epsilon_pred = model(image, noise, t, target)

        loss = model.loss(epsilon_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())

    mean_loss = float(np.mean(losses))
    print(f"► Epoch {epoch:02d} | Average Loss = {mean_loss:.4f}")
    return mean_loss

def sample(model,epoch):
    model.eval()
    with torch.no_grad():
        samples = model.sample()*0.5+0.5
        samples.clamp_(0., 1.)
        grid_image = utils.make_grid(samples, nrow=10, padding=2)
        grid_image = grid_image.numpy().transpose((1, 2, 0))  # Convert from CxHxW to HxWxC
        plt.imshow(grid_image, cmap='gray')
        plt.axis('off')  # Hide the axes

        # Sample saving handling
        timestamp = time.strftime('%Y.%m.%d-%H.%M.%S')
        config_dir =f'./samples/MNIST_{args.batch_size}_epochs{args.epochs}_lr{args.lr}/'
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
            print(f"Created directory: {config_dir}")

        # Save the image
        image_filename = config_dir + f'epoch{epoch}_{timestamp}.png'
        # filename = './samples/MNIST_' +str(epoch)+'.png'
        plt.savefig(image_filename, bbox_inches='tight', pad_inches=0)


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform  = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])])  # [0,1] to [-1,1]

    trainset = torchvision.datasets.MNIST(root='./data/MNIST',
        train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,
        batch_size=args.batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.MNIST(root='./data/MNIST',
        train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset,
        batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = DDPM(device=device).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr)
    
    # --- Training Loop ---
    print("\n🧪 Starting training...\n")
    losses = []
    total_train_time = 0
    total_sample_time = 0

    for epoch in range(args.epochs):
        # Train
        start_time = time.time()
        loss = train(model,trainloader,optimizer,epoch,device)
        end_time = time.time()
        train_time = end_time - start_time
        total_train_time += train_time
        
        # Sample
        start_time = time.time()
        sample(model,epoch)
        end_time = time.time()
        sample_time = end_time - start_time
        total_sample_time += sample_time
        
        # Log the loss
        losses.append(loss)
        print(f"Epoch {epoch:02d} | Average Loss: {loss:.4f} | Train time: {train_time:.1f}s | Sample time: {sample_time:.1f}s")

    # --- Final Analytics ---
    print("\n📊 Final Analytics:")
    print(f"Total training time: {total_train_time:.2f} sec")
    print(f"Total sampling time: {total_sample_time:.2f} sec")
    print(f"Total run time:      {total_train_time + total_sample_time:.2f} sec")
    print(f"Final loss:          {losses[-1]:.6f}")

    # Plot the losses
    plt.figure(figsize=(8,5))
    plt.plot(losses, label='Loss')
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss Over Epochs")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_loss.png")
    plt.show()

    # Save the model and results
    timestamp = time.strftime('%Y.%m.%d-%H.%M.%S')
    model_filename = f'./models/{args.batch_size}_epoch{args.epochs}_{timestamp}.pt'
    torch.save(model.state_dict(), model_filename)

    with open(f'./logs/{args.batch_size}_epoch{args.epochs}_{timestamp}_loss.pkl', 'wb') as f:
        pickle.dump({'train_loss': losses}, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--batch_size',
                        help='number of images in a mini-batch.',
                        type=int,
                        default=128)
    parser.add_argument('--epochs',
                        help='maximum number of iterations.',
                        type=int,
                        default=30)
    parser.add_argument('--lr',
                        help='initial learning rate.',
                        type=float,
                        default=1e-3)

    args = parser.parse_args()
    main(args)
