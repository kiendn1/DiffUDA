import glob
import numpy as np
import torch

# a = np.load("/home/user/code/SWG/Predictions/DAPL/OH/ViT/a2c_target_42.npy")
a = torch.load('/home/user/code/DAPrompt/output/predictions/a2c.pt')

c = 0
index = 0
count = 0
for i in sorted(glob.glob('/home/user/code/diffuda/office_home/clipart/*')):
    for j in sorted(glob.glob(i+'/*')):
        # if np.argmax(a[index]) == c:
        if torch.argmax(a[index]) == c:
            count += 1
        index += 1
    c += 1
print(count/index)

from PIL import Image
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from utils.folder import ImageFolder
from torchvision.transforms.functional import InterpolationMode
from torchvision import datasets, transforms

path = '/home/user/code/diffuda/office_home/clipart/Alarm_Clock/00040.jpg'
mean,std = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)

image = Image.open(path).convert('RGB')

transform = {
            'train': transforms.Compose(
                [
                    transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.RandomResizedCrop(size=(224, 224), scale=(0.08, 1.0), interpolation=InterpolationMode.BICUBIC),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean,std)]),
            'test': transforms.Compose(
                [
                    transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
                    transforms.CenterCrop((224,224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean,std)])
        }

u = transform['test'](image)
print(u[0][0][:30])