import torch
from diffusers import StableDiffusion3Pipeline
import glob
import os
import random

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", 
                                                torch_dtype=torch.bfloat16,
                                                token='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')
pipe = pipe.to("cuda:2")

# maps = {
#     'batteries': 'battery',
#     'candles': 'candle',
#     'clipboards': 'clipboard',
#     'flowers': 'flower',
#     'knives': 'knife',
#     'postit notes': 'postit note',
#     'sneakers': 'sneaker',
#     'toys': 'toy'
# }

list_label = [i.split('/')[-1] for i in sorted(glob.glob('/home/user/Chonnam_2023/Hung/dataset/DomainNet/painting/*'))]
for label in list_label[240:]:
    count = 1
    os.makedirs('/home/user/code/DiffUDA/images/DomainNet/stable-diffusion/painting/'+label, exist_ok=True)
    name_class = label.replace('_', ' ').lower()
    print(name_class)
    # options = ['graphic', 'cartoonish', 'outlined']
    # options = ['paintings', 'sketches',  'ornamentation', 'artistic depictions']
    for i in range(200):
        # if name_class == 'mouse':
        #     name_obj = 'computer mouse'
        # elif name_class in maps:
        #     random_float = random.uniform(0, 1)
        #     if random_float < 0.5:
        #         name_obj = maps[name_class]
        #     else:
        #         name_obj = name_class
        # else:
        #     name_obj = name_class
        name_obj = name_class

        # adj = random.choice(options)
        image = pipe(
            "artistic depictions of "+ name_obj+ " in the form of paintings",
            num_inference_steps=40,
            guidance_scale=4.5,
        ).images[0]
        image.save('/home/user/code/DiffUDA/images/DomainNet/stable-diffusion/painting/'+label+'/'+str(count)+".png")
        count += 1