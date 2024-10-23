
def dice_overlap(y_pred, y_real):
    pred = y_pred.contiguous().view(-1)
    target = y_real.contiguous().view(-1)
    
    intersection = (pred * target).sum()

    dice = (2. * intersection) / (pred.sum() + target.sum())
    
    return dice.item()  

# Intersection over Union (IoU)
def IoU(y_pred, y_real, epsilon=1e-6):

    pred = y_pred.contiguous().view(-1)
    target = y_real.contiguous().view(-1)
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection

    iou = (intersection + epsilon) / (union + epsilon)
    
    return iou.item() 

def accuracy(y_pred, y_real):

    pred = y_pred.contiguous().view(-1)
    target = y_real.contiguous().view(-1)

    TP = ((pred == 1) & (target == 1)).sum().float()
    accuracy = TP / target.numel()

    return accuracy.item()


def sensitivity(y_pred, y_real, epsilon=1e-6):

    pred = y_pred.contiguous().view(-1)
    target = y_real.contiguous().view(-1)

    TP = ((pred == 1) & (target == 1)).sum().float()
    FN = ((pred == 0) & (target == 1)).sum().float()

    sensitivity = TP / (TP + FN + epsilon)
    
    return sensitivity.item()  

def specificity(y_pred, y_real, epsilon=1e-6):
    
    pred = y_pred.contiguous().view(-1)
    target = y_real.contiguous().view(-1)

    TN = ((pred == 0) & (target == 0)).sum().float()
    FP = ((pred == 1) & (target == 0)).sum().float()

    specificity = TN / (TN + FP + epsilon)
    
    return specificity.item()  