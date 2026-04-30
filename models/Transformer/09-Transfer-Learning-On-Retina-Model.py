import os
import torch, gc
from torch import nn
from torchvision import transforms
from peft import LoraConfig, get_peft_model
from bitsandbytes.nn import Linear4bit

from pathlib import Path
import ml_modules
from timeit import default_timer as timer
import models_vit


start_time  = timer()

##################################   Clear cache    ##################################

torch.cuda.empty_cache()

gc.collect()



##################################   CONFIG    ##################################

train_dir = Path('../../dataset/augmented-data/train')
validation_dir = Path('../../dataset/augmented-data/validation')

# train_dir = Path('../../dataset/aptos2019/train')
# validation_dir = Path('../../dataset/aptos2019/validation')
BATCH_SIZE = 4
NUM_WORKERS = 8
epoch = 10
IMG_SIZE = 224
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

    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(degrees=15),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

validation_transform = transforms.Compose([

    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    normalize
])


##################################   DATALOADERS    ##################################

        
train_dataloader, vlaidation_dataloader, classes =  ml_modules.get_dataloaders(train_dir=train_dir,
                                                                test_dir=validation_dir,
                                                                BATCH_SIZE=BATCH_SIZE,
                                                                NUM_WORKERS=NUM_WORKERS,
                                                                create_sampler=False,
                                                                shuffle=True,
                                                                train_transform=minority_transform,
                                                                test_transform=validation_transform)


## model

model = models_vit.__dict__['RETFound_mae'](
    num_classes = len(classes)
)

checkpoint = torch.load("Pre-Trained-Models/RETFound_mae_natureCFP.pth", map_location='cpu', weights_only=False)
model.load_state_dict(checkpoint['model'], strict=False)
del checkpoint


## QLoRA

named_modules_dict = dict(model.named_modules())

for name, module in model.named_modules():

    if isinstance(module, nn.Linear):

        name = name.rsplit('.', 1)

        if len(name) == 1:

            parent_name = ''
            child_name = name[0]

        else:

            parent_name = name[0]
            child_name = name[-1]

        parent = model if not parent_name else named_modules_dict[parent_name]

    
        ## Create 4-bit replacement and swap

        new_layer = Linear4bit(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            quant_type='nf4',
            compute_dtype=torch.bfloat16
        )

        setattr(parent, child_name, new_layer)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["qkv"],
    lora_dropout=0.05,
    bias="none"
)

model = get_peft_model(model, lora_config)

print(f"{model.print_trainable_parameters()}")

# for params in model.parameters():

#     params.requires_grad = False

# for params in model.head.parameters():

#     params.requires_grad = True

# Total_parameters = sum((p.numel() for p in model.parameters()))
# Trainable_parameters = sum((p.numel() for p in model.parameters() if p.requires_grad))

# print(f" {Trainable_parameters}/{Total_parameters}")


experiment += 1

print(f"[INFO] Experiment : {experiment}")

## Loss function

loss_fn = nn.CrossEntropyLoss()

## Optimizer

optimizer = torch.optim.AdamW(params=filter(lambda p: p.requires_grad, model.parameters()),
                            lr=0.001)

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
                model_name='RETFound',
                extra=''
                )

MODEL_SAVE_PATH = Path("trained models")
MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)

ml_modules.save_model(model=model,
                    target_dir=MODEL_SAVE_PATH, # type: ignore
                    model_name=f"{experiment + 7}_RETFound")


terminal_width = os.get_terminal_size().columns
print("-"*terminal_width)


del model, optimizer, loss_fn

end_time  = timer()


ml_modules.print_train_time(start=start_time,
                            end=end_time,
                            device=device)