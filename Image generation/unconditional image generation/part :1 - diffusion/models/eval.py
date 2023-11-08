import os
import torch
from diffusers import DDPMPipeline

def evaluate(config, epoch, pipeline):
   
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)

    list_images = pipeline(
        batch_size = config.eval_batch_size,
        generator=torch.manual_seed(config.seed),
    ).images

    for i, image in enumerate(list_images):
        image.save(f"{test_dir}/{epoch:04d}_{i:02d}_rgb.png")


def generate(pretrained_pipe_dir):

    pipeline = DDPMPipeline.from_pretrained().to("cuda")
    images = pipeline(
        batch_size=1,
        generator=torch.manual_seed(123),
    ).images

    return images 



if __name__ == "__main__":

    generate("pretrained_pipe_dir") # replace with your model path