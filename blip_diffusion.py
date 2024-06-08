# # Alpha-CLIP in BLIP-Diffusion
# > Beyond BLIP-Diffusioin, this demonstration also works when using other project in LAVIS, like BLIP-2
# ## Prepare Environment
# You need to prepare [LAVIS](https://github.com/salesforce/LAVIS) environment first to prepare for [BLIP-Diffusion](https://github.com/salesforce/LAVIS/tree/main/projects/blip-diffusion) model, than run this notebook under LAVIS environment.

import gradio as gr
import torch
import collections
from PIL import Image
from lavis.models import load_model_and_preprocess
from torchvision import transforms
import types
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import copy 
import warnings
warnings.filterwarnings("ignore")
seed = 98765

os.environ["no_proxy"] = "localhost"
#mask_torch = None
mask_transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Resize((224, 224)),
    transforms.Normalize(0.5, 0.26)
])

alpha = None # global alpha var as alpha input
# device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
device = "cpu"
# torch.cuda.set_device("cuda:0")
# torch.cuda.set_device("cpu")
model, vis_preprocess, txt_preprocess = load_model_and_preprocess("blip_diffusion", "base", device=device, is_eval=True)


# ## Using original CLIP
# We test a image with two dog, by directly input this image into BLIP-Diffusion, it will generate a dog that is the mixtural of two dogs of different species.

class ImageMask(gr.components.Image):
    """
    Sets: source="canvas", tool="sketch"
    """

    is_template = True
    def __init__(self, **kwargs):
        super().__init__(source="upload", tool="sketch", interactive=True, **kwargs)

    def preprocess(self, x):
        return super().preprocess(x)


#negative_prompt = "over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate"
#pth = "demo_imgs/diffusion_images"
#raw_image = Image.open(pth + "/" + "image.png").convert("RGB")
# display(raw_image.resize((256, 256)))
#raw_image.save('raw_image.png')

"""
image = vis_preprocess["eval"](raw_image).unsqueeze(0).to(device)
con_subject = ""
cond_subject = con_subject
tgt_subject = con_subject

text_prompt = ""
ori_text = text_prompt

cond_subjects = [txt_preprocess["eval"](cond_subject)]
tgt_subjects = [txt_preprocess["eval"](tgt_subject)]
text_prompt = [txt_preprocess["eval"](text_prompt)]

cond_images = image
samples = {
    "cond_images": cond_images,
    "cond_subject": cond_subjects,
    "tgt_subject": tgt_subjects,
    "prompt": text_prompt,
}
num_output = 1
"""
# derek: toggle this
# output = model.generate(
#                             samples,
#                             seed=seed,
#                             guidance_scale=7.5,
#                             num_inference_steps=50,
#                             neg_prompt=negative_prompt,
#                             height=512,
#                             width=512,
#                         )
# display(output[0])
# output[0].save('output.png')


# ## Plugin Alpha-CLIP
# Alpha-CLIP can replace orginal CLIP used in BLIP-Diffusion. for simplicity, we rewrite forward funcation of its visual encoder. this rewrited_forward use alpha conv layer to add alpha-map into CLIP model input.



def rewrited_forward(self, x: torch.Tensor):
    global alpha
    if alpha is None: # better 
        print(f"[Warning] in {type(self)} forward: no alpha input when use alpha CLIP, alpha is expected!")
        alpha = torch.ones_like((x[:, [0], :, :])) * 1.9231
    x = self.conv1(x)  # shape = [*, width, grid, grid]
    x = x + self.conv1_alpha(alpha)
    x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
    x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
    x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
    x = x + self.positional_embedding.to(x.dtype)
    x = self.ln_pre(x)

    x = x.permute(1, 0, 2)  # NLD -> LND
    x = self.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD

    return x


# Then, register rewrited forward function to replace original forward function of CLIP model used in BLIP-Diffusion. change its weight into Alpha-CLIP model weight.

ori_visual_encoder = copy.deepcopy(model.blip.visual_encoder)

state_dict = torch.load('../../checkpoints/clip_l14_grit+mim_fultune_6xe.pth')
converted_dict = collections.OrderedDict()
for k, v in state_dict.items():
    # if "visual" in k:
    if 'in_proj.weight' in k:
        converted_dict[k.replace('in_proj.weight', 'in_proj_weight')] = v
    elif 'in_proj.bias' in k:
        converted_dict[k.replace('in_proj.bias', 'in_proj_bias')] = v
    else:
        converted_dict[k] = v

model.blip.visual_encoder.conv1_alpha = torch.nn.Conv2d(in_channels=1,
                                                    out_channels=model.blip.visual_encoder.conv1.out_channels, 
                                                    kernel_size=model.blip.visual_encoder.conv1.kernel_size, 
                                                    stride=model.blip.visual_encoder.conv1.stride, 
                                                    bias=False)
model.blip.visual_encoder.forward = types.MethodType(rewrited_forward, model.blip.visual_encoder)
model.blip.visual_encoder.load_state_dict(converted_dict, strict=False)
new_visual_encoder = copy.deepcopy(model.blip.visual_encoder)

# model.blip.visual_encoder = model.blip.visual_encoder.half().cuda()
# model.blip.visual_encoder = model.blip.visual_encoder.half()


# After steps above, Alpha-CLIP successfully replaces original CLIP, and can perform region focused image variation.

@torch.no_grad()
def main(
    input_im,
    task,
    focus_scale=1.0,
    scale=3.0,
    n_samples=4,
    steps=25,
    seed=0,
    text="",
    tgt=""
    ):
  
    if "Ori CLIP" == task: # different samplers
        model.blip.visual_encoder = ori_visual_encoder
    else:
        model.blip.visual_encoder = new_visual_encoder
    #generator = torch.Generator(device=device).manual_seed(int(seed))
    '''
    tform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(
        (224, 224),
        interpolation=transforms.InterpolationMode.BICUBIC,
        antialias=False,
        ),
        transforms.Normalize(
          [0.48145466, 0.4578275, 0.40821073],
          [0.26862954, 0.26130258, 0.27577711]),
    ])

    inp = tform(input_im['image']).to(device)
    '''
    mask_ori = input_im['mask'] # different mask strategy
    mask = np.array(mask_ori)[:,:,0:1]
    
    # original mask visualization
    from PIL import Image
    if "With mask" == task or "No mask" == task:
        input_mask = mask[:, :, 0]
        filename = "input_mask.png"
        print(f'Input mask has been saved as {filename}')
        im = Image.fromarray(input_mask)
        im.save(filename)
    
    if "With mask" == task: # highlight
        mask = (mask[:, :, 0] > 0)
        mask = mask * focus_scale
        mask += (255 - focus_scale)
    if "No mask" == task: # All one
        mask = (mask[:, :, 0] > -1) * 255
    
    # binary mask visualization
    if "With mask" == task or "No mask" == task:
        filename = "binary_mask.png"
        print(f'Binary mask has been saved as {filename}')
        im = Image.fromarray(mask.astype(np.uint8))
        im.save(filename)

    #    mask = mask_transform((mask * 255).astype(np.uint8))
    mask = mask_transform((mask).astype(np.uint8))
    #mask = mask.cuda().unsqueeze(dim=0)
    mask = mask.unsqueeze(dim=0)
    global alpha
    alpha = mask
    
    negative_prompt = "over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate"
    image = vis_preprocess["eval"](input_im['image']).unsqueeze(0).to(device)
    con_subject = tgt
    cond_subject = con_subject
    tgt_subject = con_subject

    text_prompt = text
    ori_text = text_prompt

    cond_subjects = [txt_preprocess["eval"](cond_subject)]
    tgt_subjects = [txt_preprocess["eval"](tgt_subject)]
    text_prompt = [txt_preprocess["eval"](text_prompt)]

    cond_images = image
    samples = {
        "cond_images": cond_images,
        "cond_subject": cond_subjects,
        "tgt_subject": tgt_subjects,
        "prompt": text_prompt,
    }
    images=[]
    for i in range(n_samples):
        image = model.generate(
                            samples,
                            seed=seed+i,
                            guidance_scale=scale,
                            num_inference_steps=steps,
                            neg_prompt=negative_prompt,
                            height=512,
                            width=512,
                        )
        images.append(image[0])
    return images

inputs = [
    ImageMask(label="[Stroke] Draw on Image",type="pil"), gr.inputs.Radio(choices=["With mask", "No mask", "Ori CLIP"], type="value", label="Interative Mode"),
    gr.Slider(0, 255, value=255, step=1, label="Focus scale"),
    gr.Slider(0, 25, value=3, step=1, label="Guidance scale"),
    gr.Slider(1, 4, value=1, step=1, label="Number images"),
    gr.Slider(5, 50, value=25, step=5, label="Steps"),
    gr.Number(0, label="Seed", precision=0),
    gr.Textbox(label="prompt"),
    gr.Textbox(label="target")
]
output = gr.Gallery(label="Generated variations")
output.style(grid=2)

examples = [
    ["example2.JPEG", "Ori CLIP", 255, 3, 1, 15, 0, "", ""],
    ["example3.JPEG", "Ori CLIP", 255, 3, 1, 15, 0, "", ""],
    ["example4.JPEG", "Ori CLIP", 255, 3, 1, 15, 0, "", ""],
    ["example5.jpg", "Ori CLIP", 255, 3, 1, 15, 0, "", ""],
]

article = \
"""
Conditional Subject: subject in conditional image.
Target Subject : desired outcome.
in this case Conditional Subject = Target Subject.
"""

with gr.Blocks() as demo:
    title_markdown = ("""
    # 🌟 [2D Image Varation] Alpha-CLIP with Blip-Diffusion            
    """)
    gr.Markdown(title_markdown)
    gr.Interface(
        fn=main,
        article=article,
        inputs=inputs,
        outputs=output,
        examples=examples,
        )
demo.queue().launch(share=True)





'''
output = model.generate(
                            samples,
                            seed=seed,
                            guidance_scale=7.5,
                            num_inference_steps=50,
                            neg_prompt=negative_prompt,
                            height=512,
                            width=512,
                        )
# display(output[0])
output[0].save('output2.png')
'''
