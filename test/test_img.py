import torch
import matplotlib.pyplot as plt
from setup import config
from models.build import build_models
from utils.data_loader import build_loader
import pandas as pd
import argparse

def denormalize(image, mean, std):
    """
    Denormalize an image tensor using the given mean and standard deviation.
    
    Args:
        image (torch.Tensor): Normalized image tensor of shape [C, H, W]
        mean (list): Mean values for each channel
        std (list): Standard deviation values for each channel
    
    Returns:
        torch.Tensor: Denormalized image tensor
    """
    image = image.clone()
    for t, m, s in zip(image, mean, std):
        t.mul_(s).add_(m)
    return image

def main():
    _, test_loader, num_classes, _, _, _ = build_loader(config)
    
    model = build_models(config, num_classes)
    
    argparser = argparse.ArgumentParser(description='Test the model')
    argparser.add_argument('--weights', type=str, help='Path to the model weights')
    args = argparser.parse_args()
    weight_path = args.weights
    checkpoint = torch.load(weight_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    class_file = 'test/oxford_flower_102_name.csv'
    
    class_df = pd.read_csv(class_file)

    class_indices = class_df['Index'].tolist()
    class_names = class_df['Name'].tolist()
    
    print(f"Number of classes: {len(class_indices)}")
    assert len(class_indices) == 102
    imgs = 0
    x_list, y_list = [], []

    for xn, yn in test_loader:
        x_list.append(xn)
        y_list.append(yn)
        imgs += len(xn)

        if imgs >= 12:
            break

    x = torch.cat(x_list, dim=0)[:12].to(device)
    y = torch.cat(y_list, dim=0)[:12]
    
    with torch.no_grad():
        output = model(x)
        if isinstance(output, (list, tuple)):
            logits = output[0]
        else:
            logits = output
        predicted = torch.argmax(logits, dim=1).cpu()
        


    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    images = [denormalize(img.cpu(), mean, std) for img in x]
    
    images = [img.permute(1, 2, 0).numpy() for img in images]

    predicted_labels = [class_indices[p.item()] for p in predicted]
    actual_labels = [class_indices[label.item()] for label in y]

    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
        ax.set_title(f"Predicted: {class_names[predicted_labels[i]]}\nActual: {class_names[actual_labels[i]]}")

        ax.add_patch(plt.Rectangle((0, 0), images[i].shape[1], images[i].shape[0], 
                           linewidth=5, edgecolor='green' if predicted_labels[i] == actual_labels[i] else 'red', facecolor='none'))
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('figures/test_images.png')
    plt.close()

if __name__ == '__main__':
    main()