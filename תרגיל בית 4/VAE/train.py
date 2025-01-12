"""Training procedure for NICE.
"""

import argparse
import torch, torchvision
from torchvision import transforms
import numpy as np
from VAE import Model

def train(vae, trainloader, optimizer, epoch):
    vae.train()  # set to training mode
    
    tot_loss = 0
    for batch_idx, (x, _) in enumerate(trainloader):
        x = x.to(vae.device)                  # Moves the input images to the correct device
        recon, mu, logvar = vae(x)            # Forward pass
        loss = vae.loss(x, recon, mu, logvar) # Compute the loss
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track total loss
        tot_loss += loss.item()

        # Log progress every 100 batches
        if batch_idx % 100 == 0: 
            print(f"Epoch {epoch}, Batch {batch_idx}/{len(trainloader)}: Loss = {loss.item(): .4f}")
        
    # Compute average loss for the epoch
    avg_loss = tot_loss / len(trainloader)
    print(f"Epoch {epoch}: Average Training Loss: {avg_loss: .4f}")
    return avg_loss


def test(vae, testloader, filename, epoch):
    vae.eval()  # set to inference mode
    
    tot_loss = 0
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(testloader):
            x = x.to(vae.device)
            recon, mu, logvar = vae(x)            # Forward pass
            loss = vae.loss(x, recon, mu, logvar) # Compute the loss
        
            # Track total loss
            tot_loss += loss.item()

            # Save samples from the first batch
            if batch_idx == 0: 
                sample_images = recon[:100] # Takes the first 100 images
                torchvision.utils.save_image(sample_images, f"{filename}_epoch{epoch}_.png", nrow=4, normalize=True)

    # Compute average loss for the test set
    avg_loss = tot_loss / len(testloader)
    print(f"Epoch {epoch}: Average Test Loss: {avg_loss: .4f}")
    return avg_loss

# Define the custom transformation outside of the main function
def dequantize(x):
    return x + torch.zeros_like(x).uniform_(0., 1. / 256.)

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        test_loss  = test(vae, trainloader, optimizer, epoch)

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

    parser.add_argument('--latent-dim',
                        help='.',
                        type=int,
                        default=100)
    parser.add_argument('--lr',
                        help='initial learning rate.',
                        type=float,
                        default=1e-3)

    args = parser.parse_args()
    main(args)
