import torch
from torch.utils.data import SequentialSampler, DataLoader
from torchvision.models import alexnet
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode

import sys
import os.path as osp
file_path = osp.dirname(osp.abspath(__file__))
sys.path.append(osp.join(file_path, '../../'))

from pytorch2tikz import Architecure

class ClassificationPresetEval:
    def __init__(
        self,
        *,
        crop_size,
        resize_size=256,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
    ):

        self.transforms = transforms.Compose(
            [
                transforms.Resize(resize_size, interpolation=interpolation),
                transforms.CenterCrop(crop_size),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('Load model')
    model = alexnet(True)
    model.eval()
    model.to(device)

    print('Load data')
    val_resize_size, val_crop_size = 256, 224
    interpolation = InterpolationMode('bilinear')
    
    preprocessing = ClassificationPresetEval(
        crop_size=val_crop_size, resize_size=val_resize_size, interpolation=interpolation
    )

    dataset_test = datasets.ImageFolder(
        osp.abspath(osp.join(file_path, '..', 'image_net')),
        preprocessing.transforms,
    )

    
    test_sampler = SequentialSampler(dataset_test)

    data_loader = DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, pin_memory=True
    )
    
    print('Init architecture')
    arch = Architecure(model, image_path=osp.join(file_path, 'input_{i}.jpg'), height_depth_factor=0.5, width_factor=0.5)

    print('Run model')
    with torch.inference_mode():
        for image, _ in data_loader:
            image = image.to(device, non_blocking=True)
            print(image.size())
            output = model(image)
    
    print('Final architecture', arch)

    out_path = osp.join(file_path, 'out.tex')
    print('Write result to out.tex')
    arch.save(out_path)
    
