import torch
from diffusers import FluxImg2ImgPipeline
from PIL import Image
import glob
import os
import random

import torch
from diffusers import FluxImg2ImgPipeline
from PIL import Image
pipe = FluxImg2ImgPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16, token='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', 
                                           device_map='balanced',
                                           max_memory={0:"24GB", 2:"24GB", 3:"1GB", 1:"1GB"})

list_label = [i.split('/')[-1] for i in sorted(glob.glob('/home/user/Chonnam_2023/Hung/dataset/Office Home/Art/*'))]
options = ['paintings', 'sketches',  'ornamentation', 'artistic depictions']
for label in list_label[32:]:
    count = 1
    os.makedirs('/home/user/code/DiffUDA/images/flux/R2A/'+label, exist_ok=True)
    name_class = label.replace('_', ' ').lower()
    if name_class == 'mouse':
        name_obj = 'computer mouse'
    print(name_class)
    for img_path in sorted(glob.glob('/home/user/Chonnam_2023/Hung/dataset/Office Home/Real/'+label+'/*')):
        adj = random.choice(options)
        init_image = Image.open(img_path).convert("RGB")
        prompt = "a " + name_class + " with " + adj +" art style"
        images = pipe(prompt, image=init_image, guidance_scale=4.5, num_inference_steps=50, strength=0.865, height=768, width=1024)
        images.images[0].save('/home/user/code/DiffUDA/images/flux/R2A/'+label+'/'+str(count)+".png")
        count += 1
        images = pipe(prompt, image=init_image, guidance_scale=4.5, num_inference_steps=50, strength=0.9, height=768, width=1024)
        images.images[0].save('/home/user/code/DiffUDA/images/flux/R2A/'+label+'/'+str(count)+".png")
        count += 1