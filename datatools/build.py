
from torchvision import transforms
from torchvision.transforms import InterpolationMode
mapping = {
    'IRRA': (384,128),
    'RaSa': (384,384),
    'APTM': (384,128)
}
def build_transforms(model_name):
    height, width = mapping[model_name]
    if model_name == 'IRRA':
        transform = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            #T.Normalize(mean=mean, std=std),
        ])
    elif model_name == 'RaSa':
        transform = transforms.Compose([
                transforms.Resize((height, width),interpolation=InterpolationMode.BICUBIC),#interpolation=InterpolationMode.BICUBIC 会影响最终的结果，但并不大
                transforms.ToTensor(),
                #T.Normalize(mean=mean, std=std),
        ])
    elif model_name == 'APTM':
        transform = transforms.Compose([
        transforms.Resize((height, width), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])
    return transform
def build_transforms_infer(model_name):
    height, width = mapping[model_name]
    transform = transforms.Compose([
            transforms.Resize((height, width),interpolation=InterpolationMode.BICUBIC),
    ])
    return transform