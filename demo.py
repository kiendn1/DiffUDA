from diffusers import AutoencoderKL
from pipeline_demofusion_sdxl import DemoFusionSDXLPipeline
import torch, gc
from torchvision import transforms
from PIL import Image

def load_and_process_image(pil_image):
    transform = transforms.Compose(
        [
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    image = transform(pil_image)
    image = image.unsqueeze(0).half()
    return image


def pad_image(image):
    w, h = image.size
    if w == h:
        return image
    elif w > h:
        new_image = Image.new(image.mode, (w, w), (0, 0, 0))
        pad_w = 0
        pad_h = (w - h) // 2
        new_image.paste(image, (0, pad_h))
        return new_image
    else:
        new_image = Image.new(image.mode, (h, h), (0, 0, 0))
        pad_w = (h - w) // 2
        pad_h = 0
        new_image.paste(image, (pad_w, 0))
        return new_image

def generate_images(prompt, negative_prompt, height, width, num_inference_steps, guidance_scale, cosine_scale_1, cosine_scale_2, cosine_scale_3, sigma, view_batch_size, stride, seed, input_image):
    padded_image = pad_image(input_image).resize((1024, 1024)).convert("RGB")
    image_lr = load_and_process_image(padded_image).to('cuda')
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = DemoFusionSDXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", vae=vae, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    generator = torch.Generator(device='cuda')
    generator = generator.manual_seed(int(seed))
    images = pipe(prompt, negative_prompt=negative_prompt, generator=generator,
                  height=int(height), width=int(width), view_batch_size=int(view_batch_size), stride=int(stride),
                  num_inference_steps=int(num_inference_steps), guidance_scale=guidance_scale,
                  cosine_scale_1=cosine_scale_1, cosine_scale_2=cosine_scale_2, cosine_scale_3=cosine_scale_3, sigma=sigma,
                  multi_decoder=True, show_image=False, lowvram=False, image_lr=image_lr
                 )
    for i, image in enumerate(images):
      image.save('image_'+str(i)+'.png')
    pipe = None
    gc.collect()
    torch.cuda.empty_cache()
    return (images[0], images[-1])


o = generate_images('clipart style', '', 500, 500, num_inference_steps=50, guidance_scale=7.5,
              cosine_scale_1=3, cosine_scale_2=1, cosine_scale_3=1, sigma=0.8,
              view_batch_size=1, stride=64, seed=42, input_image=Image.open('/home/user/code/DiffUDA/Office Home/Art/Alarm_Clock/00001.jpg'))

print(o[0])