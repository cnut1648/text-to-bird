import argparse, os
from diffusers import StableDiffusionPipeline
from typing import List
from PIL import Image
import numpy as np
from torchmetrics.image.fid import FrechetInceptionDistance
from einops import rearrange
from cub_data import CUB_coarse, CUB_fine


import torch
import asyncio
from tqdm.auto import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("-n", default=2, type=int)
    parser.add_argument("-t", default="coarse", choices=["coarse", "fine"])
    return parser.parse_args()

def batchify(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

if __name__ == "__main__":
    args = parse_args()
    
    #### dataset
    if args.t == "coarse":
        raw_datasets = CUB_coarse()["test"]
    else: # fine
        raw_datasets = CUB_fine()["test"]
    
    #### model
    pipeline = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)
    if args.ckpt is not None:
        print("Loading ckpt from ", args.ckpt)
        pipeline.unet.load_attn_procs(args.ckpt)
    else:
        print("Using vanilla stable diffusion 2")
    
    pipeline.enable_xformers_memory_efficient_attention()
    pipeline.unet = torch.compile(pipeline.unet)
    pipeline.to("cuda")


    #### eval
    save_dir = os.path.join("output", "stabilityai/stable-diffusion-2-1", "" if args.ckpt is None else f"ft_lora_{args.t}", "eval", args.t if args.ckpt is None else "")
    fids = []
    with torch.no_grad():
        all_classes = range(len(raw_datasets.features["label"].names)) # int label
        # all_classes = range(100)
        for class_ in all_classes:
            print(class_)
            fid = FrechetInceptionDistance(feature=2048)
            class_dir = os.path.join(save_dir, str(class_))
            os.makedirs(class_dir, exist_ok=True)
            # for real
            # filter raw_datasets
            class_dataset = raw_datasets.filter(lambda x: x["label"] == class_)
            real_imgs = []
            for real_img in class_dataset["image"]:
                real_img_ = torch.from_numpy(np.array(real_img.resize((256, 256)))) # should be (256, 256, 3)
                if real_img_.shape == (256, 256):
                    # repeat to 3 channels
                    real_img_ = torch.stack([real_img_, real_img_, real_img_], dim=-1)
                real_imgs.append(real_img_)
            real_imgs = torch.stack(real_imgs).to(torch.uint8)
            real_imgs = rearrange(real_imgs, "b h w c -> b c h w")
            fid.update(real_imgs, real=True)
            
            # for synthetic
            prompts = class_dataset["caption"][:5]
            generated = []
            for prompt_chunk in batchify(prompts, args.n):
                generated_chunk = pipeline(prompt_chunk, num_inference_steps=50).images
                generated += generated_chunk
            for gen in generated:
                cur_i = len(list(os.listdir(class_dir))) + 1
                gen.resize((256, 256)).save(os.path.join(class_dir, f"{cur_i}.png"))
            generated_imgs = torch.stack([
                torch.from_numpy(np.array(gen))
                for gen in generated
            ])
            generated_imgs = rearrange(generated_imgs, "b h w c -> b c h w")
            fid.update(generated_imgs, real=False)

            score = fid.compute().item()
            with open(os.path.join(save_dir, f"{class_}.fid={score}.txt"), "w") as f:
                f.write(str(score))
            fids.append(score)
    print(np.mean(fids))