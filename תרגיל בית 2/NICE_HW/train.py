"""Training procedure for NICE.
"""

import argparse
import torch
import torchvision
from torchvision import transforms
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import time
import os
import nice


def train(flow, trainloader, optimizer, epoch):
    flow.train()  # set to training mode
    tot_nll     = 0
    for inputs,_ in tqdm(trainloader):
        inputs =  inputs.view(inputs.shape[0],inputs.shape[1]*inputs.shape[2]*inputs.shape[3]) #change  shape from BxCxHxW to Bx(C*H*W)
        inputs = inputs.to(flow.device) # Moves the inputs to the correct device
        optimizer.zero_grad()           # Sets the gradiant to zero after every iteration
        nll    = -flow(inputs).mean()   # Compute the negative log-likekihood
        nll.backward()                  # Appling Backpropagation
        optimizer.step()                # Gradient Descent step

        # Track total NLL
        tot_nll += nll.item() 

    nll_avg = tot_nll / len(trainloader)
    print(f"Epoch {epoch}: Average Training Negative Log-Likelihhod: {nll_avg}")
    print(f"Epoch {epoch}: Training completed.")
    return nll_avg

def test(flow, testloader, filename, epoch, sample_shape):
    flow.eval()  # set to inference mode
    with torch.no_grad():
        tot_nll     = 0
        samples = flow.sample(100).cpu()
        a,b = samples.min(), samples.max()
        samples = (samples-a)/(b-a+1e-10) 
        samples = samples.view(-1,sample_shape[0],sample_shape[1],sample_shape[2])
        timestamp = time.strftime('%Y.%m.%d-%H.%M.%S')

        config_dir =f'./samples/{args.dataset}_{args.batch_size}_{args.coupling}_{args.coupling_type}_{args.mid_dim}_{args.hidden}/'
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
            print(f"Created directory: {config_dir}")

        image_filename = config_dir + f'{args.dataset}_{args.batch_size}_{args.coupling}_{args.coupling_type}_{args.mid_dim}_{args.hidden}_epoch{epoch}_{timestamp}.png'
        torchvision.utils.save_image(torchvision.utils.make_grid(samples),
                                     image_filename)
        
        # Compute negative log-likelihood
        for inputs, _ in tqdm(testloader):
            inputs = inputs.view(inputs.shape[0],inputs.shape[1]*inputs.shape[2]*inputs.shape[3]) #change  shape from BxCxHxW to Bx(C*H*W) 
            inputs = inputs.to(flow.device) # Moves the inputs to the correct device
            nll          = -flow(inputs).mean().item()

            # Track total NLL
            tot_nll     += nll

        nll_avg = tot_nll / len(testloader)
        print(f"Epoch {epoch}: Average Testing Negative Log-Likelihood: {nll_avg}")
    return nll_avg

# Define the custom transformation outside of the main function
def dequantize(x):
    return x + torch.zeros_like(x).uniform_(0., 1. / 256.)


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sample_shape = [1,28,28]
    full_dim = sample_shape[0]*sample_shape[1]*sample_shape[2]
    transform  = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (1.,)),
        transforms.Lambda(dequantize) #dequantization
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

    model_save_filename = '%s_' % args.dataset \
             + 'batch%d_' % args.batch_size \
             + 'coupling%d_' % args.coupling \
             + 'coupling_type%s_' % args.coupling_type \
             + 'mid%d_' % args.mid_dim \
             + 'hidden%d_' % args.hidden \
             + '.pt'

    flow = nice.NICE(
                prior=args.prior,
                coupling=args.coupling,
                coupling_type=args.coupling_type,
                in_out_dim=full_dim, 
                mid_dim=args.mid_dim,
                hidden=args.hidden,
                device=device).to(device)

    optimizer = torch.optim.Adam(
        flow.parameters(), lr=args.lr)

    # Train and test the model
    train_nlls = []
    test_nlls  = []
    for epoch in range(1, args.epochs + 1):
        train_nll = train(flow, trainloader, optimizer, epoch)
        test_nll  = test(flow, testloader, model_save_filename, epoch, sample_shape)
        train_nlls.append(train_nll)
        test_nlls.append(test_nll)
    
    #Plot the losses
    epochs = range(1, args.epochs + 1)
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Plot train losses on the primary y-axis
    ax1.plot(epochs, train_nlls, label="Train Loss", color="blue")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Train Loss", color="blue")
    ax1.tick_params(axis='y', labelcolor="blue")

    # Create a secondary y-axis for test losses
    ax2 = ax1.twinx()
    ax2.plot(epochs, test_nlls, label="Test Loss", color="red")
    ax2.set_ylabel("Test Loss", color="red")
    ax2.tick_params(axis='y', labelcolor="red")

    # Add a title and legend
    plt.title("Train and Test Loss Over Epochs")
    fig.tight_layout()
    plt.show()

    # Save the model and the results
    timestamp = time.strftime('%Y.%m.%d-%H.%M.%S')
    model_filename = f'./models/{args.dataset}_{args.batch_size}_{args.coupling}_{args.coupling_type}_{args.mid_dim}_{args.hidden}_epoch{epoch}_{timestamp}.pt'

    torch.save(flow.state_dict(),model_filename)
    with open(f'./logs/{args.dataset}_{args.batch_size}_{args.coupling}_{args.coupling_type}_{args.mid_dim}_{args.hidden}_epoch{epoch}_{timestamp}_nll.pkl', 'wb') as f:
        pickle.dump({'train_nll': train_nlls, 'test_nll': test_nlls}, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--dataset',
                        help='dataset to be modeled.',
                        type=str,
                        default='mnist')
    parser.add_argument('--prior',
                        help='latent distribution.',
                        type=str,
                        default='logistic')
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
    parser.add_argument('--coupling-type',
                        help='.',
                        type=str,
                        default='additive')
    parser.add_argument('--coupling',
                        help='.',
                        # type=int,
                        default=4)
    parser.add_argument('--mid-dim',
                        help='.',
                        type=int,
                        default=1000)
    parser.add_argument('--hidden',
                        help='.',
                        type=int,
                        default=5)
    parser.add_argument('--lr',
                        help='initial learning rate.',
                        type=float,
                        default=1e-3)

    args = parser.parse_args()
    main(args)
