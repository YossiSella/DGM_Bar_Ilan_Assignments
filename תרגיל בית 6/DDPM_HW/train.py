

import argparse
import torch, torchvision
from torchvision import transforms, utils
import numpy as np
from DDPM import DDPM
import matplotlib.pyplot as plt
from tqdm import tqdm

def train(model, trainloader, optimizer, epoch,device):

    progress_bar = tqdm(trainloader, total=len(trainloader), desc=f"Epoch {epoch:02d}")
    losses = []

    model.train()  # set to training mode
    for image, target in progress_bar:
        noise  = torch.randn_like(image).to(device)
        image  = image.to(device)
        target = target.to(device)
        
        t = torch.randint(low=0, high=model.timesteps, size=(image.size(0),), device=device).long

        epsilon_pred = model(image, noise, t, target)

        loss = model.loss(epsilon_pred, noise)


        loss = model.loss(epsilon_pred, noise)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())

    mean_loss = float(np.mean(losses))
    print(f"► Epoch {epoch:02d} | loss = {mean_loss:.4f}")
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
        filename = './samples/MNIST_' +str(epoch)+'.png'
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)


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
    for i in range(args.epochs):
        train(model,trainloader,optimizer,i,device)
        sample(model,i)

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
