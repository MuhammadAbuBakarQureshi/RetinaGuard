import torch
import os
from torch import nn
from torchvision import transforms, models

from pathlib import Path
import ml_modules
from timeit import default_timer as timer 


# def main():

start_time = timer()


##################################   CONFIG    ##################################

train_dir = Path('../../dataset/aptos2019/train')
validation_dir = Path('../../dataset/aptos2019/validation')
BATCH_SIZE = 32
NUM_WORKERS = 0

epochs = [5] 
models_name = ["resnet_101"]
transforms_options = ["manual", "auto"]

iter_experiments = 0
total_experiments = len(epochs) * len(models_name) * len(transforms_options) 


##################################   DEVICE    ##################################

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"[INFO] Using device: {device}")


##################################   NORMALIZATION    ##################################

## Normalization

normalize = transforms.Normalize(

    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

##################################   MANUAL TRANSFORM    ##################################

## Manual and simple transforms

manual_transform = transforms.Compose([
    
    transforms.Resize((224, 224)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31,
                                interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    normalize
])

##################################   AUTO TRANSFORM    ##################################

## effnet_b2_auto_transform

effnet_b2_transform = models.EfficientNet_B2_Weights.DEFAULT.transforms()


## resnet_101_auto_transforms

resnet_101_transform = models.ResNet101_Weights.DEFAULT.transforms()







##################################   DATALOADERS    ##################################

train_dataloader, vlaidation_dataloader, classes =  ml_modules.get_dataloaders(train_dir=train_dir,
                                                                        test_dir=validation_dir,
                                                                        BATCH_SIZE=BATCH_SIZE,
                                                                        NUM_WORKERS=NUM_WORKERS,
                                                                        train_transform=manual_transform,
                                                                        test_transform=manual_transform)






##################################   EXPERIMENTS    ##################################


for epoch in epochs:

    for model_name in models_name:

        match model_name:

            case "effnet_b2":

                model = ml_modules.create_effnetb2(num_classes=len(classes))

            case "resnet_101":

                model = ml_modules.create_resnet101(num_classes=len(classes))
            
            case _:

                assert "Model name mismatch in model switch block"

        for transform_name in transforms_options:

            iter_experiments +=1

            print(f"[INFO] Experiment : {iter_experiments} / {total_experiments}")

            print(f"[INFO] Epochs: {epoch}")

            print(f"[INFO] Model: {model_name}")

            print(f"[INFO] Transform: {transform_name}")

            if transform_name == "auto":

                match model_name:

                    case "effnet_b2":

                        transform = effnet_b2_transform

                    case "resnet_101":

                        transform = resnet_101_transform
                    
                    case _:

                        assert "Model name mismatch in transform switch block"

            else:

                transform = manual_transform

        
            train_dataloader, vlaidation_dataloader, classes =  ml_modules.get_dataloaders(train_dir=train_dir,
                                                                            test_dir=validation_dir,
                                                                            BATCH_SIZE=BATCH_SIZE,
                                                                            NUM_WORKERS=NUM_WORKERS,
                                                                            train_transform=transform,
                                                                            test_transform=transform)



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
                            experiment_name=f"{transform_name}_transform_{epoch}_epochs",
                            model_name=model_name,
                            extra='CNN'
                            )

            MODEL_SAVE_PATH = Path("trained models")
            MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)

            ml_modules.save_model(model=model,
                                target_dir=MODEL_SAVE_PATH, # type: ignore
                                model_name=model_name)
            

            terminal_width = os.get_terminal_size().columns
            print("-"*terminal_width)

            
end_time = timer()

ml_modules.print_train_time(start=start_time,
                            end=end_time,
                            device=device)

# if __name__ == "__main__":

#     main()