
from torchvision import transforms

normalize = transforms.Normalize(

    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

minority_transform = transforms.Compose([

    transforms.Resize((224, 224)),
    transforms.RandomRotation(degrees=15),
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

majority_transform = transforms.Compose([

    transforms.Resize((224, 224)),
    transforms.RandomRotation(degrees=5),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

validation_transform = transforms.Compose([

    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])
