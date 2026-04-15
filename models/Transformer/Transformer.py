import torch
import os
from torch import nn
from torchvision import transforms, models

from pathlib import Path
import ml_modules
from timeit import default_timer as timer 


# def main():


##################################   CONFIG    ##################################

train_dir = Path('../../dataset/aptos2019/train')
validation_dir = Path('../../dataset/aptos2019/validation')
BATCH_SIZE = 32
NUM_WORKERS = 8

epoch = 5 


##################################   DEVICE    ##################################

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"[INFO] Using device: {device}")

##################################   AUTO TRANSFORM    ##################################


# ImageClassification(
#     crop_size=[224]
#     resize_size=[256]
#     interpolation=InterpolationMode.BILINEAR
# )

normalize = transforms.Normalize(

    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

transform = transforms.Compose([
    
    transforms.Resize((224, 224)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31,
                            interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    normalize
])


# transform = models.ViT_B_16_Weights.DEFAULT.transforms()

##################################   DATALOADERS    ##################################

        
train_dataloader, vlaidation_dataloader, classes =  ml_modules.get_dataloaders(train_dir=train_dir,
                                                                test_dir=validation_dir,
                                                                BATCH_SIZE=BATCH_SIZE,
                                                                NUM_WORKERS=NUM_WORKERS,
                                                                train_transform=transform,
                                                                test_transform=transform)


## model

model = ml_modules.create_vit_b_16(len(classes))

## Loss function

loss_fn = nn.CrossEntropyLoss()

## Optimizer

optimizer = torch.optim.SGD(params=model.parameters(),
                            lr=0.01)


ml_modules.fit_fn(model=model,
                train_dataloader=train_dataloader,
                test_dataloader=vlaidation_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                classes=classes,
                batch_size=BATCH_SIZE,
                epochs=epoch,
                device=device,
                experiment_name=f"{epoch}_epochs",
                model_name='vit_b_16',
                extra='Transformer'
                )

MODEL_SAVE_PATH = Path("trained models")
MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)

ml_modules.save_model(model=model,
                    target_dir=MODEL_SAVE_PATH, # type: ignore
                    model_name='2_vit_b_16')


terminal_width = os.get_terminal_size().columns
print("-"*terminal_width)

# if __name__ == "__main__":

#     main()