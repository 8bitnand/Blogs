import os
import torch


def evaluate(condition_images, config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`

    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)

    list_images = pipeline(
        condition_images=condition_images,
        batch_size=config.eval_batch_size,
        output_type="pil",
        generator=torch.manual_seed(config.seed),
    ).images
