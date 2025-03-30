import torch
import matplotlib.pyplot as plt
from setup import config
from models.build import build_models
from utils.data_loader import build_loader
import pandas as pd
from tqdm import tqdm
from prettytable import PrettyTable

def denormalize(image, mean, std):
    image = image.clone()
    for t, m, s in zip(image, mean, std):
        t.mul_(s).add_(m)
    return image

def evaluate_model():
    _, test_loader, num_classes, _, _, _ = build_loader(config)
    model = build_models(config, num_classes)
    weight_path = 'output/checkpoint.bin'
    checkpoint = torch.load(weight_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    table = PrettyTable()
    class_file = 'test/oxford_flower_102_name.csv'
    class_df = pd.read_csv(class_file)
    class_names = class_df['Index'].tolist()
    
    correct_top1, correct_top5, total = 0, 0, 0
    
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc='Evaluating', unit='batch', dynamic_ncols=True)
        for x, y in progress_bar:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            
            top1_pred = torch.argmax(logits, dim=1)
            top5_pred = torch.topk(logits, 5, dim=1).indices
            
            correct_top1 += (top1_pred == y).sum().item()
            correct_top5 += sum(y[i] in top5_pred[i] for i in range(y.size(0)))
            total += y.size(0)
            
            progress_bar.set_description(f"Evaluating (Top-1: {100 * correct_top1 / total:.2f}%, Top-5: {100 * correct_top5 / total:.2f}%)")
            
    
    accuracy_top1 = 100 * correct_top1 / total
    accuracy_top5 = 100 * correct_top5 / total
    
    table.field_names = ["Metric", "Value"]
    table.add_row(["Top-1 Accuracy (%)", f"{accuracy_top1:.2f}"])
    table.add_row(["Top-5 Accuracy (%)", f"{accuracy_top5:.2f}"])
    
    print(table)

if __name__ == '__main__':
    evaluate_model()
