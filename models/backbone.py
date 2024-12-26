import torch
import torch.nn as nn
import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

import torch
import torch.nn as nn

from dassl.engine import TRAINER_REGISTRY, TrainerXU
from dassl.metrics import compute_accuracy
from dassl.utils import MetricMeter, AverageMeter, load_pretrained_weights, load_checkpoint, save_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

class CLIP(nn.Module):
    def __init__(self,args):
        super(CLIP, self).__init__()
        if args.model_name == "RN50":
            model = load_clip_to_cpu("RN50")
            self.output_num = 1024
        elif args.model_name == "RN101":
            model = load_clip_to_cpu("RN101")
            self.output_num = 512
        elif args.model_name == "VIT-B":
            model = load_clip_to_cpu("ViT-B/16")
            self.output_num = 512
        if args.datasets=="office_home":
            class_list = ['an image of a alarm clock', 'an image of a backpack', 'an image of a batteries', 'an image of a bed', 'an image of a bike', 'an image of a bottle', 'an image of a bucket', 'an image of a calculator', 'an image of a calendar', 'an image of a candles', 'an image of a chair', 'an image of a clipboards', 'an image of a computer', 'an image of a couch', 'an image of a curtains', 'an image of a desk lamp', 'an image of a drill', 'an image of a eraser', 'an image of a exit sign', 'an image of a fan', 'an image of a file cabinet', 'an image of a flipflops', 'an image of a flowers', 'an image of a folder', 'an image of a fork', 'an image of a glasses', 'an image of a hammer', 'an image of a helmet', 'an image of a kettle', 'an image of a keyboard', 'an image of a knives', 'an image of a lamp shade', 'an image of a laptop', 'an image of a marker', 'an image of a monitor', 'an image of a mop', 'an image of a mouse', 'an image of a mug', 'an image of a notebook', 'an image of a oven', 'an image of a pan', 'an image of a paper clip', 'an image of a pen', 'an image of a pencil', 'an image of a postit notes', 'an image of a printer', 'an image of a push pin', 'an image of a radio', 'an image of a refrigerator', 'an image of a ruler', 'an image of a scissors', 'an image of a screwdriver', 'an image of a shelf', 'an image of a sink', 'an image of a sneakers', 'an image of a soda', 'an image of a speaker', 'an image of a spoon', 'an image of a tv', 'an image of a table', 'an image of a telephone', 'an image of a toothbrush', 'an image of a toys', 'an image of a trash can', 'an image of a webcam']
        elif args.datasets=="visda":
            class_list = ['a photo of a aeroplane', 'a photo of a bicycle', 'a photo of a bus', 'a photo of a car', 'a photo of a horse', 'a photo of a knife', 'a photo of a motorcycle', 'a photo of a person', 'a photo of a plant', 'a photo of a skateboard', 'a photo of a train', 'a photo of a truck']
        elif args.datasets=="digits":
            class_list = ['a photo of the number: "0"','a photo of the number: "1"','a photo of the number: "2"','a photo of the number: "3"','a photo of the number: "4"','a photo of the number: "5"','a photo of the number: "6"','a photo of the number: "7"','a photo of the number: "8"','a photo of the number: "9"']
        elif args.datasets == "office31":
            class_list = ['a photo of a backpack', 'a photo of a bike', 'a photo of a bike helmet', 'a photo of a bookcase', 'a photo of a bottle', 'a photo of a calculator', 'a photo of a desk chair', 'a photo of a desk lamp', 'a photo of a desktop computer', 'a photo of a file cabinet', 'a photo of a headphones', 'a photo of a keyboard', 'a photo of a laptop computer', 'a photo of a letter tray', 'a photo of a mobile phone', 'a photo of a monitor', 'a photo of a mouse', 'a photo of a mug', 'a photo of a paper notebook', 'a photo of a pen', 'a photo of a phone', 'a photo of a printer', 'a photo of a projector', 'a photo of a punchers', 'a photo of a ring binder', 'a photo of a ruler', 'a photo of a scissors', 'a photo of a speaker', 'a photo of a stapler', 'a photo of a tape dispenser', 'a photo of a trash can']
        elif args.datasets=="domain_net":
            class_list= ['a photo of a aircraft carrier', 'a photo of a airplane', 'a photo of a alarm clock', 'a photo of a ambulance', 'a photo of a angel', 'a photo of a animal migration', 'a photo of a ant', 'a photo of a anvil', 'a photo of a apple', 'a photo of a arm', 'a photo of a asparagus', 'a photo of a axe', 'a photo of a backpack', 'a photo of a banana', 'a photo of a bandage', 'a photo of a barn', 'a photo of a baseball', 'a photo of a baseball bat', 'a photo of a basket', 'a photo of a basketball', 'a photo of a bat', 'a photo of a bathtub', 'a photo of a beach', 'a photo of a bear', 'a photo of a beard', 'a photo of a bed', 'a photo of a bee', 'a photo of a belt', 'a photo of a bench', 'a photo of a bicycle', 'a photo of a binoculars', 'a photo of a bird', 'a photo of a birthday cake', 'a photo of a blackberry', 'a photo of a blueberry', 'a photo of a book', 'a photo of a boomerang', 'a photo of a bottlecap', 'a photo of a bowtie', 'a photo of a bracelet', 'a photo of a brain', 'a photo of a bread', 'a photo of a bridge', 'a photo of a broccoli', 'a photo of a broom', 'a photo of a bucket', 'a photo of a bulldozer', 'a photo of a bus', 'a photo of a bush', 'a photo of a butterfly', 'a photo of a cactus', 'a photo of a cake', 'a photo of a calculator', 'a photo of a calendar', 'a photo of a camel', 'a photo of a camera', 'a photo of a camouflage', 'a photo of a campfire', 'a photo of a candle', 'a photo of a cannon', 'a photo of a canoe', 'a photo of a car', 'a photo of a carrot', 'a photo of a castle', 'a photo of a cat', 'a photo of a ceiling fan', 'a photo of a cell phone', 'a photo of a cello', 'a photo of a chair', 'a photo of a chandelier', 'a photo of a church', 'a photo of a circle', 'a photo of a clarinet', 'a photo of a clock', 'a photo of a cloud', 'a photo of a coffee cup', 'a photo of a compass', 'a photo of a computer', 'a photo of a cookie', 'a photo of a cooler', 'a photo of a couch', 'a photo of a cow', 'a photo of a crab', 'a photo of a crayon', 'a photo of a crocodile', 'a photo of a crown', 'a photo of a cruise ship', 'a photo of a cup', 'a photo of a diamond', 'a photo of a dishwasher', 'a photo of a diving board', 'a photo of a dog', 'a photo of a dolphin', 'a photo of a donut', 'a photo of a door', 'a photo of a dragon', 'a photo of a dresser', 'a photo of a drill', 'a photo of a drums', 'a photo of a duck', 'a photo of a dumbbell', 'a photo of a ear', 'a photo of a elbow', 'a photo of a elephant', 'a photo of a envelope', 'a photo of a eraser', 'a photo of a eye', 'a photo of a eyeglasses', 'a photo of a face', 'a photo of a fan', 'a photo of a feather', 'a photo of a fence', 'a photo of a finger', 'a photo of a fire hydrant', 'a photo of a fireplace', 'a photo of a firetruck', 'a photo of a fish', 'a photo of a flamingo', 'a photo of a flashlight', 'a photo of a flip flops', 'a photo of a floor lamp', 'a photo of a flower', 'a photo of a flying saucer', 'a photo of a foot', 'a photo of a fork', 'a photo of a frog', 'a photo of a frying pan', 'a photo of a garden', 'a photo of a garden hose', 'a photo of a giraffe', 'a photo of a goatee', 'a photo of a golf club', 'a photo of a grapes', 'a photo of a grass', 'a photo of a guitar', 'a photo of a hamburger', 'a photo of a hammer', 'a photo of a hand', 'a photo of a harp', 'a photo of a hat', 'a photo of a headphones', 'a photo of a hedgehog', 'a photo of a helicopter', 'a photo of a helmet', 'a photo of a hexagon', 'a photo of a hockey puck', 'a photo of a hockey stick', 'a photo of a horse', 'a photo of a hospital', 'a photo of a hot air balloon', 'a photo of a hot dog', 'a photo of a hot tub', 'a photo of a hourglass', 'a photo of a house', 'a photo of a house plant', 'a photo of a hurricane', 'a photo of a ice cream', 'a photo of a jacket', 'a photo of a jail', 'a photo of a kangaroo', 'a photo of a key', 'a photo of a keyboard', 'a photo of a knee', 'a photo of a knife', 'a photo of a ladder', 'a photo of a lantern', 'a photo of a laptop', 'a photo of a leaf', 'a photo of a leg', 'a photo of a light bulb', 'a photo of a lighter', 'a photo of a lighthouse', 'a photo of a lightning', 'a photo of a line', 'a photo of a lion', 'a photo of a lipstick', 'a photo of a lobster', 'a photo of a lollipop', 'a photo of a mailbox', 'a photo of a map', 'a photo of a marker', 'a photo of a matches', 'a photo of a megaphone', 'a photo of a mermaid', 'a photo of a microphone', 'a photo of a microwave', 'a photo of a monkey', 'a photo of a moon', 'a photo of a mosquito', 'a photo of a motorbike', 'a photo of a mountain', 'a photo of a mouse', 'a photo of a moustache', 'a photo of a mouth', 'a photo of a mug', 'a photo of a mushroom', 'a photo of a nail', 'a photo of a necklace', 'a photo of a nose', 'a photo of a ocean', 'a photo of a octagon', 'a photo of a octopus', 'a photo of a onion', 'a photo of a oven', 'a photo of a owl', 'a photo of a paint can', 'a photo of a paintbrush', 'a photo of a palm tree', 'a photo of a panda', 'a photo of a pants', 'a photo of a paper clip', 'a photo of a parachute', 'a photo of a parrot', 'a photo of a passport', 'a photo of a peanut', 'a photo of a pear', 'a photo of a peas', 'a photo of a pencil', 'a photo of a penguin', 'a photo of a piano', 'a photo of a pickup truck', 'a photo of a picture frame', 'a photo of a pig', 'a photo of a pillow', 'a photo of a pineapple', 'a photo of a pizza', 'a photo of a pliers', 'a photo of a police car', 'a photo of a pond', 'a photo of a pool', 'a photo of a popsicle', 'a photo of a postcard', 'a photo of a potato', 'a photo of a power outlet', 'a photo of a purse', 'a photo of a rabbit', 'a photo of a raccoon', 'a photo of a radio', 'a photo of a rain', 'a photo of a rainbow', 'a photo of a rake', 'a photo of a remote control', 'a photo of a rhinoceros', 'a photo of a rifle', 'a photo of a river', 'a photo of a roller coaster', 'a photo of a rollerskates', 'a photo of a sailboat', 'a photo of a sandwich', 'a photo of a saw', 'a photo of a saxophone', 'a photo of a school bus', 'a photo of a scissors', 'a photo of a scorpion', 'a photo of a screwdriver', 'a photo of a sea turtle', 'a photo of a see saw', 'a photo of a shark', 'a photo of a sheep', 'a photo of a shoe', 'a photo of a shorts', 'a photo of a shovel', 'a photo of a sink', 'a photo of a skateboard', 'a photo of a skull', 'a photo of a skyscraper', 'a photo of a sleeping bag', 'a photo of a smiley face', 'a photo of a snail', 'a photo of a snake', 'a photo of a snorkel', 'a photo of a snowflake', 'a photo of a snowman', 'a photo of a soccer ball', 'a photo of a sock', 'a photo of a speedboat', 'a photo of a spider', 'a photo of a spoon', 'a photo of a spreadsheet', 'a photo of a square', 'a photo of a squiggle', 'a photo of a squirrel', 'a photo of a stairs', 'a photo of a star', 'a photo of a steak', 'a photo of a stereo', 'a photo of a stethoscope', 'a photo of a stitches', 'a photo of a stop sign', 'a photo of a stove', 'a photo of a strawberry', 'a photo of a streetlight', 'a photo of a string bean', 'a photo of a submarine', 'a photo of a suitcase', 'a photo of a sun', 'a photo of a swan', 'a photo of a sweater', 'a photo of a swing set', 'a photo of a sword', 'a photo of a syringe', 'a photo of a t-shirt', 'a photo of a table', 'a photo of a teapot', 'a photo of a teddy-bear', 'a photo of a telephone', 'a photo of a television', 'a photo of a tennis racquet', 'a photo of a tent', 'a photo of a the eiffel tower', 'a photo of a the great wall of china', 'a photo of a the mona lisa', 'a photo of a tiger', 'a photo of a toaster', 'a photo of a toe', 'a photo of a toilet', 'a photo of a tooth', 'a photo of a toothbrush', 'a photo of a toothpaste', 'a photo of a tornado', 'a photo of a tractor', 'a photo of a traffic light', 'a photo of a train', 'a photo of a tree', 'a photo of a triangle', 'a photo of a trombone', 'a photo of a truck', 'a photo of a trumpet', 'a photo of a umbrella', 'a photo of a underwear', 'a photo of a van', 'a photo of a vase', 'a photo of a violin', 'a photo of a washing machine', 'a photo of a watermelon', 'a photo of a waterslide', 'a photo of a whale', 'a photo of a wheel', 'a photo of a windmill', 'a photo of a wine bottle', 'a photo of a wine glass', 'a photo of a wristwatch', 'a photo of a yoga', 'a photo of a zebra', 'a photo of a zigzag']
        elif args.datasets=="image_clef":
            class_list = ['a photo of a aeroplane', 'a photo of a bike', 'a photo of a bird', 'a photo of a boat', 'a photo of a bottle', 'a photo of a bus', 'a photo of a car', 'a photo of a dog', 'a photo of a horse', 'a photo of a monitor', 'a photo of a motorbike', 'a photo of a people']

        self.model = model
        self.args = args
        self.text = clip.tokenize(class_list)
        text_features = self.encode_text().detach()
        self.text_features = text_features / text_features.norm(dim=1, keepdim=True)
            
    def forward_features(self, x):
        feature = self.model.encode_image(x)
        return feature

    def encode_text(self):
        text_features = self.model.encode_text(self.text)
        return text_features

    def forward_head(self,image_features, return_text_logit=False, src_image=None):
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = self.text_features
        # cosine similarity as logits
        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()        
        logits_per_text = logits_per_image.t()
        if return_text_logit:
            return logits_per_image,logits_per_text
        else:
            return logits_per_image

    def forward(self, x, src_image=None):
        image_features = self.forward_features(x)
        logits_per_image = self.forward_head(image_features)

        return logits_per_image

def load_clip_to_cpu(backbone_name):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, '/kaggle/output/models')
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model

def get_text_features(args, model_path, backbone_name, classnames):
    #Load checkpoint
    checkpoint = load_checkpoint(model_path)
    state_dict = checkpoint["state_dict"]
    # Ignore fixed token vectors
    if "token_prefix" in state_dict:
        del state_dict["token_prefix"]

    if "token_suffix" in state_dict:
        del state_dict["token_suffix"]
    ctx_vectors = state_dict['ctx'].cpu()
    domain_vectors = state_dict['domain_vectors'].cpu()

    n_dm = 2
    n_dmx = 16
    n_ctx = 16
    n_cls = len(classnames)
    n = n_dmx + n_ctx

    clip_model = load_clip_to_cpu(backbone_name)
    dtype = clip_model.dtype

    domainnames = [args.src_domain.lower(),  args.tgt_domain.lower()]
    domainnames = [
        ", a {} image.".format(domain.replace('real', 'real_world')) for domain in domainnames
    ]
    prompt_prefix = " ".join(["X"] * 32)
    prompts = [
        prompt_prefix + " " + name + " " + domain + "."
        for domain in domainnames for name in classnames
    ]

    naive_prompt_prefix = "a photo of a".replace("_", " ")
    naive_prompts = [
        naive_prompt_prefix + " " + name + "." for name in classnames
    ]

    print(f'Initial context: "{prompt_prefix}"')
    print(f"Number of context words (tokens): {n_ctx}")
    print(f"Number of domain context words (tokens): {n_dmx}")

    tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
    naive_tokenized_prompts = torch.cat(
        [clip.tokenize(p) for p in naive_prompts])

    with torch.no_grad():
        embedding = clip_model.token_embedding(tokenized_prompts).type(
            dtype)
        naive_embedding = clip_model.token_embedding(
            naive_tokenized_prompts).type(dtype)

    # These token vectors will be saved when in save_model(),
    # but they should be ignored in load_model() as we want to use
    # those computed using the current class names
    tokenized_prompts = torch.cat(
        [tokenized_prompts, naive_tokenized_prompts])
    token_prefix = embedding[:, :1, :]  # SOS
    token_suffix = embedding[:, 1 + n:, :]  # CLS, EOS

    ctx = ctx_vectors
    ctx_dim = ctx.size(-1)
    dmx = domain_vectors  # dm 16 512
    ctx = ctx.unsqueeze(0).expand(n_dm, -1, -1, -1)  # dm 16 512
    dmx = dmx.unsqueeze(1).expand(-1, n_cls, -1, -1)  # dm cls 16 512
    ctxdmx = torch.cat([ctx, dmx],
                    dim=2).reshape(n_cls * n_dm,
                                    n_ctx + n_dmx, ctx_dim)

    prefix = token_prefix
    suffix = token_suffix

    # naive
    neb = naive_embedding

    prompts = torch.cat(
        [
            prefix,  # (n_cls, 1, dim)
            ctxdmx,  # (n_cls, n_ctx, dim)
            suffix,  # (n_cls, *, dim)
        ],
        dim=1,
    )
    prompts = torch.cat([prompts, neb], dim=0)

    prompts = prompts.reshape(195,1,77,512)
    tokenized_prompts = tokenized_prompts.reshape(195,1,77)
    list_r = []
    with torch.no_grad():
        for i in range(195):
            x = prompts[i] + clip_model.positional_embedding.type(dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = clip_model.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = clip_model.ln_final(x).type(dtype)
            x = x[torch.arange(x.shape[0]), tokenized_prompts[i].argmax(dim=-1)] @ clip_model.text_projection
            list_r.append(x)
        text_features = torch.cat(list_r, dim=0)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features


class CustomCLIP(nn.Module):
    def __init__(self, args, classnames):
        super().__init__()

        if args.model_name == "RN50":
            model = load_clip_to_cpu("RN50")
            self.output_num = 1024
        elif args.model_name == "RN101":
            model = load_clip_to_cpu("RN101")
            self.output_num = 512
        elif args.model_name == "VIT-B":
            model = load_clip_to_cpu("ViT-B/16")
            self.output_num = 512
        
        self.model = model  
        self.args = args
        self.logit_scale = model.logit_scale
        self.dtype = model.dtype
        model_path = f'/kaggle/input/cp-dapl/cp-daprompt/{args.src_domain.lower()[0]}2{args.tgt_domain.lower()[0]}_seed42_model.pth.tar-200'
        self.text_features = get_text_features(args, model_path, "ViT-B/16", classnames)

    def forward_features(self, x):
        feature = self.model.visual(x.type(self.dtype))
        return feature

    def forward(self, image, src_image=False):
        image_features = self.model.visual(image.type(self.dtype))
        logits_per_image = self.forward_head(image_features, src_image=src_image)
        return logits_per_image
    
    def forward_head(self,image_features, return_text_logit=False, src_image=False):
        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale.to(image_features.device) * image_features @ (self.text_features.t().to(image_features.device))
        logits_per_image = logits_per_image.reshape(
            -1, 3, 65)
        if src_image:
            logits_per_image = logits_per_image[:,-1,:]
        else:
            logits_per_image = logits_per_image[:,-2,:]
        
        logits_per_text = logits_per_image.t()
        if return_text_logit:
            return logits_per_image,logits_per_text
        else:
            return logits_per_image

def get_backbone(args):
    if args.use_dapl:
        class_list = ['an image of a alarm clock', 'an image of a backpack', 'an image of a batteries', 'an image of a bed', 'an image of a bike', 'an image of a bottle', 'an image of a bucket', 'an image of a calculator', 'an image of a calendar', 'an image of a candles', 'an image of a chair', 'an image of a clipboards', 'an image of a computer', 'an image of a couch', 'an image of a curtains', 'an image of a desk lamp', 'an image of a drill', 'an image of a eraser', 'an image of a exit sign', 'an image of a fan', 'an image of a file cabinet', 'an image of a flipflops', 'an image of a flowers', 'an image of a folder', 'an image of a fork', 'an image of a glasses', 'an image of a hammer', 'an image of a helmet', 'an image of a kettle', 'an image of a keyboard', 'an image of a knives', 'an image of a lamp shade', 'an image of a laptop', 'an image of a marker', 'an image of a monitor', 'an image of a mop', 'an image of a mouse', 'an image of a mug', 'an image of a notebook', 'an image of a oven', 'an image of a pan', 'an image of a paper clip', 'an image of a pen', 'an image of a pencil', 'an image of a postit notes', 'an image of a printer', 'an image of a push pin', 'an image of a radio', 'an image of a refrigerator', 'an image of a ruler', 'an image of a scissors', 'an image of a screwdriver', 'an image of a shelf', 'an image of a sink', 'an image of a sneakers', 'an image of a soda', 'an image of a speaker', 'an image of a spoon', 'an image of a tv', 'an image of a table', 'an image of a telephone', 'an image of a toothbrush', 'an image of a toys', 'an image of a trash can', 'an image of a webcam']
        class_name = [c.replace('an image of a ', '') for c in class_list]
        model = CustomCLIP(args, class_name)
    else:
        model = CLIP(args)
    return model