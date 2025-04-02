import torch
import matplotlib.pyplot as plt
import numpy as np
from setup import config
from models.build import build_models
from utils.data_loader import build_loader
from tqdm import tqdm
import argparse

def linear_cka(X, Y):
    """Compute Linear Centered Kernel Alignment (CKA) between matrices."""
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    
    X = X / (X.norm(dim=1, keepdim=True) + 1e-8)
    Y = Y / (Y.norm(dim=1, keepdim=True) + 1e-8)
    
    xt_x = X.T @ X
    yt_y = Y.T @ Y
    xt_y = X.T @ Y
    
    dot_product = (xt_y ** 2).sum()
    norm_x = (xt_x ** 2).sum().sqrt()
    norm_y = (yt_y ** 2).sum().sqrt()
    
    return (dot_product / (norm_x * norm_y + 1e-8)).item()

def evaluate_model():
    argparser = argparse.ArgumentParser(description='')
    argparser.add_argument('--weights', type=str, help='Path to the model weights')
    argparser.add_argument('--batch_size', type=int, help='Number of forward passes (each with one sample)')
    args = argparser.parse_args()
    config.defrost()
    config.data.batch_size = 1
    config.model.forward_features = True
    config.freeze()
    
    _, test_loader, num_classes, _, _, _ = build_loader(config)
    model = build_models(config, num_classes)

    checkpoint = torch.load(args.weights, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    layer_features = []
    num_layers = None

    loader_iter = iter(test_loader)
    for _ in tqdm(range(args.batch_size), desc='Evaluating Iterations', unit='iter'):
        try:
            x, _ = next(loader_iter)
        except StopIteration:
            break

        x = x.to(device)

        with torch.no_grad():
            _, features = model(x)
        
        if num_layers is None:
            num_layers = len(features)
            layer_features = [[] for _ in range(num_layers)]
        
        for i, feat in enumerate(features):
            pooled = torch.mean(feat, dim=1)
            layer_features[i].append(pooled.cpu())

    for i in range(num_layers):
        layer_features[i] = torch.cat(layer_features[i], dim=0)

    cka_matrix = np.zeros((num_layers, num_layers))
    for i in range(num_layers):
        for j in range(num_layers):
            cka_matrix[i, j] = linear_cka(layer_features[i], layer_features[j])

    plt.figure(figsize=(10, 8))

    plt.imshow(cka_matrix, cmap='inferno', vmin=0, vmax=1, origin='lower')
    plt.colorbar(label='CKA Similarity')
    plt.title("Layer-wise CKA Similarity")
    plt.xlabel("Layer Index")
    plt.ylabel("Layer Index")
    plt.savefig("figures/cka_map.png", bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == '__main__':
    evaluate_model()
