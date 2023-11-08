# https://huggingface.co/docs/diffusers/tutorials/basic_training

import os
import glob
import torch
import math
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from datasets import load_dataset
from dataclasses import dataclass
from accelerate import Accelerator
from torchvision import transforms
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from eval import evaluate


@dataclass
class TrainingConfig:
    image_size = 128  
    train_batch_size = 8
    eval_batch_size = 8  
    num_epochs = 512
    gradient_accumulation_steps = 1
    learning_rate = 1.0e-5
    lr_warmup_steps = 500
    save_image_epochs = 16
    save_model_epochs = 16
    mixed_precision = "fp16" 
    output_dir = "m1guelpf_nouns" 
    push_to_hub = False  
    hub_private_repo = False
    overwrite_output_dir = True  
    seed = 0
    


config = TrainingConfig()

def save_plot(images):
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    for i, image in enumerate(images):
        axs[i].imshow(image)
        axs[i].set_axis_off()
    fig.show()


def transform_stc(batch):
    tfms = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    rgb_images = [tfms(image.convert("RGB")) for image in batch["image"]]

    return {"rgb": rgb_images}


def load_pipline(config):
    
    pipeline = DDPMPipeline.from_pretrained(
        "mrm8488/ddpm-ema-butterflies-128",
        cache_dir="models/pretrained",
    )
    return pipeline.unet


def train_loop(
    config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler
):
    
    accelerator = Accelerator(
        # mixed_precision=config.mixed_precision, ## might not work with accelerate 
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    for epoch in range(config.num_epochs):
        progress_bar = tqdm(
            total=len(train_dataloader), disable=not accelerator.is_local_main_process
        )
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            rgb_images = batch["rgb"]
            
            noise = torch.randn(rgb_images.shape).to(rgb_images.device)
            bs = rgb_images.shape[0]
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bs,),
                device=rgb_images.device,
            ).long()

            noisy_images = noise_scheduler.add_noise(rgb_images, noise, timesteps)

            with accelerator.accumulate(model):
            
                noise_pred = model(
                    noisy_images,
                    timesteps,
                    return_dict=False,
                )[0]

                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        if accelerator.is_main_process:
            pipeline = DDPMPipeline(
                unet=accelerator.unwrap_model(model), scheduler=noise_scheduler
            )

            if (
                epoch + 1
            ) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline)

            if (
                epoch + 1
            ) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                pipeline.save_pretrained(config.output_dir)


def main():
    config.dataset_name = "m1guelpf/nouns"
    train_dataset = load_dataset(config.dataset_name, split="train")
    train_dataset.set_transform(transform_stc)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.train_batch_size, shuffle=True
    )
    model = load_pipline(config)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )

    train_loop(
        config=config,
        model=model,
        noise_scheduler=noise_scheduler,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        lr_scheduler=lr_scheduler,
    )

if __name__ == "__main__":
    main()

