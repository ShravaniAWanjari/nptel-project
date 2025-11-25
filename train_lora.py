from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.models.attention_processor import LoRAAttnProcessor
import torch
from torch import optim
from diffusers import AutoencoderKL
from tqdm import tqdm
import os
from PIL import Image
import random
import numpy as np

DATA_FOLDER = "dataset/images"
MODEL_ID = "runwayml/stable-diffusion-v1-5"
OUTPUT_LORA = "lora/houseplan.safetensors"
EPOCHS = 8
LR = 1e-4
BATCH = 2
CAPTION = "floor plan, architectural layout, black and white blueprint, 2d plan"

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    safety_checker=None
).to("cuda")

# prevent crash on Windows
#pipe.enable_xformers_memory_efficient_attention = lambda *args, **kwargs: None

vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae").to("cuda")

lora = LoRAAttnProcessor(hidden_size=pipe.unet.config.cross_attention_dim)
pipe.unet.set_attn_processor(lora)

optimizer = optim.Adam(lora.parameters(), lr=LR)

image_files = [os.path.join(DATA_FOLDER, f) for f in os.listdir(DATA_FOLDER)]


for epoch in range(EPOCHS):
    random.shuffle(image_files)
    for i in range(0, len(image_files), BATCH):
        batch = image_files[i:i+BATCH]
        imgs = []
        for img_path in batch:
            img = Image.open(img_path).convert("RGB").resize((512, 512))
            arr = np.array(img).astype("float32") / 255.0
            arr = torch.from_numpy(arr).permute(2, 0, 1) * 2 - 1
            imgs.append(arr)

        imgs = torch.stack(imgs).to("cuda", dtype=torch.float16)
        latent = vae.encode(imgs).latent_dist.sample() * 0.18215
        prompt_embeds = pipe._encode_prompt(CAPTION, "cuda", 1, do_classifier_free_guidance=False)

        noise = torch.randn_like(latent)
        t = torch.randint(0, 1000, (latent.shape[0],), device="cuda").long()

        noisy = pipe.scheduler.add_noise(latent, noise, t)
        pred = pipe.unet(noisy, t, encoder_hidden_states=prompt_embeds).sample

        loss = torch.nn.functional.mse_loss(pred, noise)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"Epoch {epoch+1}/{EPOCHS} — Loss: {loss.item()}")

pipe.unet.save_attn_procs(OUTPUT_LORA)
print("Training complete → Saved:", OUTPUT_LORA)
