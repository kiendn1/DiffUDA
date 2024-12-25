from models.backbone import get_backbone
import glob
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

class A():
    model_name = 'VIT-B'
    datasets = "office_home"

args = A()
model = get_backbone(args)

model = model.to("cuda")

def processor(image):
    resized_image = image.resize((224, 224))

    # Define the transformation to convert the image to a tensor with shape (3, 224, 224)
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts image to a PyTorch tensor with shape (C, H, W) and scales pixel values to [0, 1]
    ])

    # Apply the transformation
    tensor_image = transform(resized_image)
    return tensor_image.unsqueeze(0)

clean_img_path = []
list_label = [i.split('/')[-1] for i in sorted(glob.glob('/home/user/code/DiffUDA/images/Office-Home/stable-diffusion/Clipart/*'))]
i = 0
for label in list_label:
    name_class = label.replace('_', ' ').lower()
    print(name_class)
    for path in sorted(glob.glob('/home/user/code/DiffUDA/images/Office-Home/stable-diffusion/Clipart/'+label+'/*')):
        image = Image.open(path)
        tensor_image = processor(image).to('cuda')
        if name_class == 'mouse':
            name_class = 'computer mouse'
        
        img_feat = model.forward_features(tensor_image)
        logits_clip = F.softmax(model.forward_head(img_feat), dim=1)
        r = logits_clip[0][i]
        if r >= 0.6:
            clean_img_path.append(path)
    print(len(clean_img_path))
    i += 1

print(len(clean_img_path))
with open("/home/user/code/DiffUDA/experiment/clean_img_path/C_06.txt", "w") as file:
    file.write("\n".join(clean_img_path) + "\n")