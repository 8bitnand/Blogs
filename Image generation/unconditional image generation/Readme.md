Running with accelerate 

```bash

# training 
cd "Image generation/unconditional image generation/part :1 - diffusion"
accelerate config
accelerate launch models/main.py 

# inferance

# replace your pretrained model/pipe path in the main 
python eval.py 
```