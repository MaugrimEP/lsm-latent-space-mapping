import torch
import torch.nn
import torch.nn.functional as F
from torchvision.transforms import transforms


# region face transforms
face_transforms_source = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda face_img: face_img/255)
])

face_transforms_target = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda y: y.flatten()),
])
# endregion

# region sarcopenia transforms
sarco_transforms_source = transforms.Compose([
    transforms.ToTensor(),
])

sarco_transforms_target = transforms.Compose([
    transforms.ToTensor(),
])
# endregion

# region sarcopenia transforms
toyset_transforms_source = transforms.Compose([
    transforms.ToTensor(),
])

toyset_transforms_target = transforms.Compose([
    transforms.ToTensor(),
])
# endregion
