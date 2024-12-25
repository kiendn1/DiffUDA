from PIL import Image
import glob
import torch

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
model = model.to("cuda")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

clean_img_path = []
list_label = [i.split('/')[-1] for i in sorted(glob.glob('/home/user/code/DiffUDA/images/Office-Home/stable-diffusion/Clipart/*'))]
for label in list_label:
    name_class = label.replace('_', ' ').lower()
    print(name_class)
    raw = []
    for path in sorted(glob.glob('/home/user/code/DiffUDA/images/Office-Home/stable-diffusion/Clipart/'+label+'/*'))[:200]:
        image = Image.open(path)
        if name_class == 'mouse':
            name_class = 'computer mouse'
        inputs = processor(text=[name_class, "not "+name_class], images=image, return_tensors="pt", padding=True)
        inputs = inputs.to("cuda")
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            print(probs)
        r = probs[0][0]
        # raw.append((r, path))
        if r >= 0.75:
            clean_img_path.append(path)
    # sorted_raw = sorted(raw, key=lambda x: x[0])
    # clean_img_path.extend([x[1] for x in sorted_raw[:180]])
    print(len(clean_img_path))

print(len(clean_img_path))
with open("/home/user/code/DiffUDA/experiment/clean_img_path/Real_75.txt", "w") as file:
    file.write("\n".join(clean_img_path) + "\n")
# class_list = ['an image of a alarm clock', 'an image of a backpack', 'an image of a batteries', 'an image of a bed', 'an image of a bike', 'an image of a bottle', 'an image of a bucket', 'an image of a calculator', 'an image of a calendar', 'an image of a candles', 'an image of a chair', 'an image of a clipboards', 'an image of a computer', 'an image of a couch', 'an image of a curtains', 'an image of a desk lamp', 'an image of a drill', 'an image of a eraser', 'an image of a exit sign', 'an image of a fan', 'an image of a file cabinet', 'an image of a flipflops', 'an image of a flowers', 'an image of a folder', 'an image of a fork', 'an image of a glasses', 'an image of a hammer', 'an image of a helmet', 'an image of a kettle', 'an image of a keyboard', 'an image of a knives', 'an image of a lamp shade', 'an image of a laptop', 'an image of a marker', 'an image of a monitor', 'an image of a mop', 'an image of a mouse', 'an image of a mug', 'an image of a notebook', 'an image of a oven', 'an image of a pan', 'an image of a paper clip', 'an image of a pen', 'an image of a pencil', 'an image of a postit notes', 'an image of a printer', 'an image of a push pin', 'an image of a radio', 'an image of a refrigerator', 'an image of a ruler', 'an image of a scissors', 'an image of a screwdriver', 'an image of a shelf', 'an image of a sink', 'an image of a sneakers', 'an image of a soda', 'an image of a speaker', 'an image of a spoon', 'an image of a tv', 'an image of a table', 'an image of a telephone', 'an image of a toothbrush', 'an image of a toys', 'an image of a trash can', 'an image of a webcam']

# image = Image.open('/home/user/code/DiffUDA/images/flux/R2C/Keyboard/54.png')
# inputs = processor(text=class_list, images=image, return_tensors="pt", padding=True)

# outputs = model(**inputs)
# logits_per_image = outputs.logits_per_image
# probs = logits_per_image.softmax(dim=1)
# print(probs)