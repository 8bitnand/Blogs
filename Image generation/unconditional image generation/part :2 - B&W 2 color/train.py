import os
import torch
import json
import numpy as np
from PIL import Image as I
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from datasets import load_dataset
from dataclasses import dataclass
from accelerate import Accelerator
from torchvision import transforms
from skimage.color import rgb2lab, lab2rgb

# from huggingface_hub import HfFolder, Repository, whoami
from diffusers import DDPMPipeline, UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from eval import evaluate


@dataclass
class TrainingConfig:
    image_size = 128  # the generated image resolution
    train_batch_size = 8
    eval_batch_size = 8  # how many images to sample during evaluation
    num_epochs = 512
    gradient_accumulation_steps = 1
    learning_rate = 3.3e-5
    lr_warmup_steps = 500
    save_image_epochs = 16
    save_model_epochs = 16
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "m1guelpf_nouns"  # the model name locally and on the HF Hub
    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0
    dataset_output_dir = "datasets/"
    dataset_name = "m1guelpf/nouns"
    model_url = "mrm8488/ddpm-ema-butterflies-128"
    model_config = "models/model_config.json"


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
            # transforms.ColorJitter(
            #     brightness=0.3, contrast=0.1, saturation=(1.0, 2.0), hue=0.05
            # ),
            transforms.ToTensor(),
        ]
    )
    rgb_images = [
        tfms(I.fromarray(rgb2lab(image.convert("RGB")).astype(np.uint8)))
        for image in batch["image"]
    ]
    gray_images = [
        tfms(I.fromarray(rgb2lab(image.convert("L").convert("RGB")).astype(np.uint8)))
        for image in batch["image"]
    ]
    return {"rgb": rgb_images, "gray": gray_images}


def load_weights(pretrained_model, uninitilized_model):
    for name, param in pretrained_model.state_dict().items():
        if param.shape == uninitilized_model.state_dict()[name].shape:
            uninitilized_model.state_dict()[name].copy_(param)

    return uninitilized_model


def load_pipline(config):
    pipeline = DDPMPipeline.from_pretrained(config.model_url)
    return pipeline


def train_loop(
    config,
    model,
    noise_scheduler,
    optimizer,
    train_dataloader,
    test_dataloader,
    lr_scheduler,
):
    accelerator = Accelerator(
        # mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    device = accelerator.device

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
            rgb_l = rgb_images[:, :1]
            rgb_ab = rgb_images[:, 1:]

            # Sample noise to add to the gray_images ab channel
            noise = torch.randn(rgb_ab.shape).to(device)
            bs = rgb_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bs,),
                device=device,
            ).long()

            noisy_images = torch.cat(
                [rgb_l, noise_scheduler.add_noise(rgb_ab, noise, timesteps)], dim=1
            )

            with accelerator.accumulate(model):
                # Predict the noise residual
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

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = DDPMPipeline(
                unet=accelerator.unwrap_model(model), scheduler=noise_scheduler
            )

            if (
                epoch + 1
            ) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                for batch in test_dataloader:
                    eval_images = batch["gray"].to(device)
                    break

                evaluate(eval_images, config, epoch, pipeline)

            if (
                epoch + 1
            ) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                pipeline.save_pretrained(config.output_dir)


def main():
    dataset = load_dataset(config.dataset_name, split="train")
    dataset = dataset.train_test_split(0.02)
    dataset.set_transform(transform_stc)

    train_dataloader = torch.utils.data.DataLoader(
        dataset["train"], batch_size=config.train_batch_size, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        dataset["test"], batch_size=config.train_batch_size, shuffle=False
    )
    pipeline = load_pipline(config)
    pretrained_model = pipeline.unet

    with open(config.model_config) as rstream:
        model_config = json.load(rstream)

    model = UNet2DModel.from_config(model_config)
    model = load_weights(pretrained_model, model)
    noise_scheduler = pipeline.scheduler

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
        test_dataloader=test_dataloader,
        lr_scheduler=lr_scheduler,
    )


# notebook_launcher(train_loop, args, num_processes=1)
if __name__ == "__main__":
    main()
