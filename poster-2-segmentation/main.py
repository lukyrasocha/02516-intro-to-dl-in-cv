import torch
import numpy as np

from torchvision import transforms

from utils.load_data import load_data
from utils.logger import logger
from utils.visualize import display_random_images_and_masks, visualize_predictions
from models.train import train_model
from models.models import EncDec, UNet 
from models.losses import bce_loss
from utils.transforms import JointTransform
from torch.utils.data import DataLoader
from models.split_image import split_image_into_patches
from utils.visualize import display_image_and_mask


PH2_TRAIN_CNN = True
DRIVE_TRAIN_CNN = True


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
logger.info(f"Running on {DEVICE}")

# Load data
logger.working_on("Loading data for PH2")

# Adjustable crop size (set to None if you don't want to crop)
CROP_SIZE = (350, 350)  # or None
RESIZE = None #(400, 400)  # Resize after cropping (if desired)

transform_ph2_train = JointTransform(crop_size=CROP_SIZE, resize=RESIZE)
transform_drive_train = JointTransform(crop_size=CROP_SIZE, resize=RESIZE)

transform_ph2 = transforms.Compose([
    transforms.Resize((400, 400)),
    transforms.ToTensor(),
])

transform_drive = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


# Load PH2 dataset with augmentation and artificial data (50 synthetic samples)
logger.working_on("Loading data for PH2")
ph2_train_dataset = load_data('ph2', split='train', transform=transform_ph2_train, crop = True)
ph2_val_dataset = load_data('ph2', split='val', transform=transform_ph2_train, crop = True)
ph2_test_dataset = load_data('ph2', split='test', transform=transform_ph2)

image, mask = ph2_train_dataset[1]
print('Image shape:', image.shape)
print('Mask shape:', mask.shape)
assert len(np.unique(mask.numpy()[0])) <= 2, "mask needs to have binary values (0,1)"

# Data loaders for PH2
ph2_train_loader = DataLoader(ph2_val_dataset, batch_size=16, shuffle=True)
ph2_test_loader = DataLoader(ph2_test_dataset, batch_size=16, shuffle=False)
ph2_val_loader = DataLoader(ph2_val_dataset, batch_size=16, shuffle=False)

# Load DRIVE dataset with augmentation for training
logger.working_on("Loading data for DRIVE")
drive_train_dataset = load_data('drive', split='train', transform=transform_drive_train, crop = True)
drive_val_dataset = load_data('drive', split='val', transform=transform_drive_train, crop = True)
drive_test_dataset = load_data('drive', split='test', transform=transform_drive)

image, mask = drive_train_dataset[1]
print('Image shape:', image.shape)
print('Mask shape:', mask.shape)
assert len(np.unique(mask.numpy()[0])) <= 2, "mask needs to have binary values (0,1)"
# Data loaders for DRIVE
drive_train_loader = DataLoader(drive_train_dataset, batch_size=3, shuffle=True)
drive_val_loader = DataLoader(drive_val_dataset, batch_size=3, shuffle=False)
drive_test_loader = DataLoader(drive_test_dataset, batch_size=3, shuffle=False)

logger.success("Data loaded")

# Display some images
display_random_images_and_masks(ph2_train_dataset, figname="ph2_random.png", num_images=3)
display_random_images_and_masks(drive_train_dataset, figname="drive_random.png", num_images=3)
logger.success("Saved example images and masks to 'figures'")


if PH2_TRAIN_CNN:
    # Simple Encoder-Decoder on PH2

    LEARNING_RATE = 0.0001
    MAX_EPOCHS = 100 
    loss_fn = bce_loss


    encdec_ph2_model = EncDec(input_channels=3, output_channels=1)
    optimizer = torch.optim.Adam(encdec_ph2_model.parameters(), lr=LEARNING_RATE)

    config= {
        "learning_rate": LEARNING_RATE,
        "architecture": "Simple-Encoder-Decoder",
        "dataset": "PH2",
        "epochs": MAX_EPOCHS,
        "loss_fn": "BinaryCrossEntropy",
        "optimizer": "Adam"
    }

    logger.working_on("Training simple Encoder-Decoder on PH2")
    train_model(encdec_ph2_model, ph2_train_loader, ph2_val_loader, loss_fn, optimizer,wandb_config=config, num_epochs=MAX_EPOCHS, device=DEVICE)
    visualize_predictions(encdec_ph2_model, ph2_train_loader, DEVICE, figname="ENCDEC_ph2_predictions.png", num_images=5)
    logger.success("Saved examples of predictions for Enc-Dec of PH2 to 'figures'")

if DRIVE_TRAIN_CNN:
    # Simple Encoder-Decoder on DRIVE
    LEARNING_RATE = 0.001
    MAX_EPOCHS = 100 
    loss_fn = bce_loss

    encdec_drive_model = EncDec(input_channels=3, output_channels=1)
    optimizer = torch.optim.Adam(encdec_drive_model.parameters(), lr=LEARNING_RATE)

    config= {
        "learning_rate": LEARNING_RATE,
        "architecture": "Simple-Encoder-Decoder",
        "dataset": "DRIVE",
        "epochs": MAX_EPOCHS,
        "loss_fn": "BinaryCrossEntropy",
        "optimizer": "Adam"
    }


    logger.working_on("Training simple Encoder-Decoder on DRIVE")
    train_model(encdec_drive_model, drive_train_loader, drive_val_loader, loss_fn, optimizer, wandb_config=config, num_epochs= MAX_EPOCHS, device=DEVICE)
    visualize_predictions(encdec_drive_model, drive_train_loader, DEVICE, figname="ENCDEC_drive_predictions.png", num_images=5)
    logger.success("Saved examples of predictions for Enc-Dec of DRIVE to 'figures'")

# Simple Encoder-Decoder on UNet
LEARNING_RATE = 0.001
MAX_EPOCHS = 100 
PADDING = 0 # no padding
loss_fn = bce_loss

UNetModel_ph2 = UNet(in_channels=3, num_classes=1, padding=PADDING)
optimizer = torch.optim.Adam(UNetModel_ph2.parameters(), lr=LEARNING_RATE)

config= {
    "learning_rate": LEARNING_RATE,
    "architecture": "UNet",
    "dataset": "PH2",
    "epochs": MAX_EPOCHS,
    "loss_fn": "BinaryCrossEntropy",
    "optimizer": "Adam",
    "padding": PADDING 
}


logger.working_on("Training simple UNet on PH2")
train_model(UNetModel_ph2, ph2_train_loader, ph2_val_loader, loss_fn, optimizer,wandb_config=config, num_epochs=MAX_EPOCHS, device=DEVICE)
visualize_predictions(UNetModel_ph2, ph2_train_loader, DEVICE, figname="UNET_ph2_predictions.png", num_images=5)
logger.success("Saved examples of predictions for UNet to 'figures'")

# Simple UNet on DRIVE
LEARNING_RATE = 0.001
MAX_EPOCHS = 100 
PADDING = 0 # no padding
loss_fn = bce_loss

UNetModel_drive = UNet(in_channels=3, num_classes=1, padding=PADDING)
optimizer = torch.optim.Adam(UNetModel_drive.parameters(), lr=LEARNING_RATE)

config= {
    "learning_rate": LEARNING_RATE,
    "architecture": "UNet",
    "dataset": "DRIVE",
    "epochs": MAX_EPOCHS,
    "loss_fn": "BinaryCrossEntropy",
    "optimizer": "Adam",
    "padding": PADDING
}

logger.working_on("Training simple UNet on DRIVE")
train_model(UNetModel_drive, drive_train_loader, drive_val_loader, loss_fn, optimizer, wandb_config=config, num_epochs= MAX_EPOCHS, device=DEVICE)
visualize_predictions(UNetModel_drive, drive_train_loader, DEVICE, figname="UNET_drive_predictions.png", num_images=5)
logger.success("Saved examples of predictions for UNet of DRIVE to 'figures'")


# Evaluation
with torch.no_grad():
    model = UNetModel_ph2.eval()

    image, mask = next(iter(ph2_test_loader))
    
    image = image.to(DEVICE)
    mask = mask.to(DEVICE)

    i = 5
    image = image[i]
    mask  = mask[i]


    print("Original image")
    print(image.shape)
    print(mask.shape)

    predicted_mask = split_image_into_patches(image, CROP_SIZE[0], UNetModel_ph2)
    #predicted_mask = split_image_into_patches(image, 256, encdec_ph2_model)

    print("Predicted mask shape")
    print(predicted_mask.shape)

    predicted_mask = predicted_mask.cpu()
    image = image.cpu()
    mask = mask.cpu()

    display_image_and_mask(image, predicted_mask, "predicted_mask_ph2.png", "1")
    display_image_and_mask(image, mask, "real_mask_ph2.png", "12")

# Plots