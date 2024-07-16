import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

def dice_coefficient(pred, target):
    smooth = 1e-5
    intersection = np.sum(pred * target)
    return (2. * intersection + smooth) / (np.sum(pred) + np.sum(target) + smooth)

def hd95(pred, target):
    pred_border = np.logical_xor(pred, binary_erosion(pred))
    target_border = np.logical_xor(target, binary_erosion(target))
    
    pred_border_dist = distance_transform_edt(~pred_border)
    target_border_dist = distance_transform_edt(~target_border)
    
    pred_distances = pred_border_dist[target_border]
    target_distances = target_border_dist[pred_border]
    
    all_distances = np.concatenate([pred_distances, target_distances])
    return np.percentile(all_distances, 95)

def evaluate_model(model, dataloader):
    model.eval()
    all_dice_whole = []
    all_dice_core = []
    all_dice_enhance = []
    all_hd95_whole = []
    all_hd95_core = []
    all_hd95_enhance = []
    
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data['image'].to(device), data['mask'].to(device)
            
            outputs = model(inputs)
            outputs = torch.argmax(outputs, dim=1).cpu().numpy()
            labels = labels.cpu().numpy()
            
            for pred, target in zip(outputs, labels):
                dice_whole = dice_coefficient(pred, target)
                dice_core = dice_coefficient(pred == 1, target == 1)
                dice_enhance = dice_coefficient(pred == 2, target == 2)
                
                hd95_whole = hd95(pred, target)
                hd95_core = hd95(pred == 1, target == 1)
                hd95_enhance = hd95(pred == 2, target == 2)
                
                all_dice_whole.append(dice_whole)
                all_dice_core.append(dice_core)
                all_dice_enhance.append(dice_enhance)
                all_hd95_whole.append(hd95_whole)
                all_hd95_core.append(hd95_core)
                all_hd95_enhance.append(hd95_enhance)
    
    metrics = {
        'dice_whole': np.mean(all_dice_whole),
        'dice_core': np.mean(all_dice_core),
        'dice_enhance': np.mean(all_dice_enhance),
        'hd95_whole': np.mean(all_hd95_whole),
        'hd95_core': np.mean(all_hd95_core),
        'hd95_enhance': np.mean(all_hd95_enhance),
    }
    
    return metrics

# Evaluate the model on the validation set
model_metrics = evaluate_model(model, val_loader)
print(f'Model Metrics: {model_metrics}')
