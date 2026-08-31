import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_inference_transform():
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
