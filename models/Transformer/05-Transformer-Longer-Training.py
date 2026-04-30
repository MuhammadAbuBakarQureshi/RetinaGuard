import os
import torch
from torch import nn
from torchvision import transforms

from pathlib import Path
import ml_modules
from timeit import default_timer as timer 


# def main():

start_time  = timer()


##################################   CONFIG    ##################################

train_dir = Path('../../dataset/aptos2019/train')
validation_dir = Path('../../dataset/aptos2019/validation')
BATCH_SIZE = 32
NUM_WORKERS = 10
models_to_train = ['vit_b_32'] 
epoch = 50
experiment = 0

##################################   DEVICE    ##################################

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"[INFO] Using device: {device}")

##################################   AUTO TRANSFORM    ##################################

normalize = transforms.Normalize(

    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

minority_transform = transforms.Compose([

    transforms.Resize((224, 224)),
    # transforms.RandomRotation(degrees=15),
    # transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomAutocontrast(),
    transforms.ToTensor(),
    normalize
])

# minority_transform = transforms.Compose([

#     transforms.Resize((224, 224)),
#     transforms.TrivialAugmentWide(num_magnitude_bins=31,
#                             interpolation=transforms.InterpolationMode.BICUBIC),
#     transforms.ToTensor(),
#     normalize
# ])



validation_transform = transforms.Compose([

    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

##################################   SAMPLER    ##################################




##################################   DATALOADERS    ##################################

        
train_dataloader, vlaidation_dataloader, classes =  ml_modules.get_dataloaders(train_dir=train_dir,
                                                                test_dir=validation_dir,
                                                                BATCH_SIZE=BATCH_SIZE,
                                                                NUM_WORKERS=NUM_WORKERS,
                                                                create_sampler=True,
                                                                shuffle=False,
                                                                train_transform=minority_transform,
                                                                test_transform=validation_transform)


## model

for model_name in models_to_train:

    experiment += 1

    print(f"[INFO] Experiment : {experiment}")

    print(f"[INFO] Model : {model_name}")

    match model_name:

        case 'vit_b_16':

            model = ml_modules.create_vit_b_16(len(classes))

        case 'vit_b_32':

            model = ml_modules.create_vit_b_32(len(classes))

    ## Loss function

    loss_fn = nn.CrossEntropyLoss()

    ## Optimizer

    optimizer = torch.optim.AdamW(params=model.parameters(),
                                lr=0.004)

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
                    model_name=model_name,
                    extra='longer_training'
                    )

    MODEL_SAVE_PATH = Path("trained models")
    MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)

    ml_modules.save_model(model=model,
                        target_dir=MODEL_SAVE_PATH, # type: ignore
                        model_name=f"{experiment + 5}_{model_name}_longer_training_{epoch}_epochs")


    terminal_width = os.get_terminal_size().columns
    print("-"*terminal_width)


end_time  = timer()


ml_modules.print_train_time(start=start_time,
                            end=end_time,
                            device=device)
# if __name__ == "__main__":

#     main()