from PIL import Image
from torchvision.utils import save_image
from torchvision import datasets, transforms
m = transforms.Compose([
                    transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.Resize([256, 256]),
                    transforms.ToTensor()
])
a = Image.open('/home/user/code/DiffUDA/images/flux/P2C/Pan/5.png')
b = m(a)
print(b.shape)

d = Image.open('/home/user/code/DiffUDA/Office Home/Product/Pan/00003.jpg')
e = m(d)

b[:,:,112:] = e[:,:,112:]

save_image(b, 'tt.png')