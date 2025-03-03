import os
import sys
import torch
import cv2
import numpy as np
import gradio as gr
from segment_anything import SamPredictor, sam_model_registry
from stable_diffusion_inpaint import replace_img_with_sd
from lama_inpaint import inpaint_img_with_lama, build_lama_model
from utils import load_img_to_array, save_array_to_img, dilate_mask

# Load SAM Model
sam_checkpoint = "./pretrained_models/sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
predictor = SamPredictor(sam)

# Load LaMa Model
lama_config = "./lama/configs/prediction/default.yaml"
lama_ckpt = "pretrained_models/big-lama"
device = "cuda" if torch.cuda.is_available() else "cpu"
lama_model = build_lama_model(lama_config, lama_ckpt, device=device)

# Function to get mask from SAM
def get_sam_mask(image, points):
    predictor.set_image(image)
    masks, _, _ = predictor.predict(point_coords=np.array(points), point_labels=np.ones(len(points)), multimask_output=False)
    return masks[0]

# Inpainting function
def inpaint_image(image, points, text_prompt=None):
    mask = get_sam_mask(image, points)
    if text_prompt:
        return replace_img_with_sd(image, mask, text_prompt, device=device)
    else:
        return inpaint_img_with_lama(lama_model, image, mask, lama_config, device=device)

# Gradio UI
def segment_and_inpaint(image, points, text_prompt):
    result = inpaint_image(image, points, text_prompt)
    return result

demo = gr.Interface(fn=segment_and_inpaint, inputs=["image", "textbox", "textbox"], outputs="image")
demo.launch()
