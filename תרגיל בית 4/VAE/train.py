"""Training procedure for NICE.
"""

import argparse
import torch, torchvision
from torchvision import transforms
import numpy as np
from VAE import Model

import time
import os
from tqdm import tqdm


def train(vae, trainloader, optimizer, epoch):
    vae.train()  # set to training mode
    
    tot_loss = 0
    progress_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc=f"Epoch {epoch}")

    for batch_idx, (x, _) in progress_bar:
        x = x.to(vae.device)                  # Moves the input images to the correct device
        recon, mu, logvar = vae(x)            # Forward pass
        loss = vae.loss(x, recon, mu, logvar) # Compute the loss
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track total loss
        tot_loss += loss.item()

        # Update progress bar
        progress_bar.set_postfix(loss=loss.item())

        # Log progress every 100 batches
        if batch_idx % 100 == 0: 
            print(f"Epoch {epoch}, Batch {batch_idx}/{len(trainloader)}: Loss = {loss.item(): .4f}")
        
    # Compute average loss for the epoch
    avg_loss = tot_loss / len(trainloader)
    print(f"Epoch {epoch}: Average Training Loss: {avg_loss: .4f}")
    print(f"Epoch {epoch}: Training completed.")
    return avg_loss


def test(vae, testloader, filename, epoch, sample_shape):
    vae.eval()  # set to inference mode
    
    tot_loss = 0
    with torch.no_grad():
        samples = vae.sample(100)
        samples = vae.upsample(samples)
        samples = samples.view(-1, 64, 7, 7)
        samples = vae.decoder(samples).cpu()
        a, b    = samples.min(), samples.max()
        samples = (samples - a) / (b - a + 1e-10)
        samples = samples.view(-1, sample_shape[0], sample_shape[1], sample_shape[2])

        # Sample saving handling
        timestamp = time.strftime('%Y.%m.%d-%H.%M.%S')
        config_dir =f'./samples/{args.dataset}_{args.batch_size}_{args.latent_dim}/'
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
            print(f"Created directory: {config_dir}")

        image_filename = config_dir + f'{args.dataset}_{args.batch_size}_{args.latent_dim}_epoch{epoch}_{timestamp}.png'
        torchvision.utils.save_image(torchvision.utils.make_grid(samples),
                                     image_filename)
     
        # Test loop
        progress_bar = tqdm(enumerate(testloader), total=len(testloader), desc=f"Test Epoch {epoch}")

        for batch_idx, (x, _) in progress_bar:
            x = x.to(vae.device)
            recon, mu, logvar = vae(x)            # Forward pass
            loss = vae.loss(x, recon, mu, logvar) # Compute the loss
        
            # Track total loss
            tot_loss += loss.item()

            # Update progress bar
            progress_bar.set_postfix(loss=loss.item())
    
        # Compute average loss for the test set
        avg_loss = tot_loss / len(testloader)
        print(f"Epoch {epoch}: Average Test Loss: {avg_loss: .4f}")
    return avg_loss

# Define the custom transformation outside of the main function
def dequantize(x):
    return x + torch.zeros_like(x).uniform_(0., 1. / 256.)

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sample_shape = [1,28,28] # mnist image shape
    transform  = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(dequantize), #dequantization
        transforms.Normalize((0.,), (257./256.,)), #rescales to [0,1]

    ])

    if args.dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root='./data/MNIST',
            train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.MNIST(root='./data/MNIST',
            train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
            batch_size=args.batch_size, shuffle=False, num_workers=2)
    elif args.dataset == 'fashion-mnist':
        trainset = torchvision.datasets.FashionMNIST(root='~/torch/data/FashionMNIST',
            train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST',
            train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
            batch_size=args.batch_size, shuffle=False, num_workers=2)
    else:
        raise ValueError('Dataset not implemented')

    filename = '%s_' % args.dataset \
             + 'batch%d_' % args.batch_size \
             + 'mid%d_' % args.latent_dim

    vae = Model(latent_dim=args.latent_dim,device=device).to(device)
    optimizer = torch.optim.Adam(
        vae.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_loss = train(vae, trainloader, optimizer, epoch)
        test_loss  = test(vae, trainloader, filename, epoch, sample_shape)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--dataset',
                        help='dataset to be modeled.',
                        type=str,
                        default='mnist')
    parser.add_argument('--batch_size',
                        help='number of images in a mini-batch.',
                        type=int,
                        default=128)
    parser.add_argument('--epochs',
                        help='maximum number of iterations.',
                        type=int,
                        default=50)
    parser.add_argument('--sample_size',
                        help='number of images to generate.',
                        type=int,
                        default=64)

    parser.add_argument('--latent_dim',
                        help='.',
                        type=int,
                        default=100)
    parser.add_argument('--lr',
                        help='initial learning rate.',
                        type=float,
                        default=1e-3)

    args = parser.parse_args()
    main(args)
