# Segment Anything Model (SAM) with LaMa Inpainting

## Overview
This project integrates **Segment Anything Model (SAM)** with **LaMa Inpainting** to enable object removal and mask-based image inpainting. Users can segment objects using **SAM** and seamlessly fill missing regions using **LaMa**.

### Optional Description
The **Segment Anything + LaMa Inpainting** project provides an automated approach to **image completion and object removal**. It combines cutting-edge deep learning models:
- **Segment Anything Model (SAM)** from Meta AI to accurately segment objects.
- **LaMa (Large Mask Inpainting)** to fill in missing regions naturally.

By leveraging **SAM's precise segmentation** and **LaMa's generative capabilities**, the system ensures **seamless object removal and image restoration**. The workflow is ideal for applications like **photo editing, scene reconstruction, and artistic modifications**.

## Dataset
- **SA-1B Dataset**: 1 billion segmentation masks from 11 million images (SAM dataset).
- **LaMa Dataset**: Large-scale dataset for learning-based image inpainting.

### Dataset Links
- [Segment Anything Dataset (SA-1B)](https://segment-anything.com)
- [LaMa Inpainting Model](https://github.com/saic-mdal/lama)

## Model Implementation
- **Segment Anything Model (SAM)** for object segmentation.
- **LaMa Model** for inpainting the segmented regions.
- **PyTorch-based implementation** for efficient inference.

## Technologies Used
- **Python**
- **PyTorch**
- **OpenCV**
- **NumPy**

## Installation & Setup
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- PyTorch & Torchvision
- OpenCV
- NumPy

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sam-lama-inpainting.git
   cd sam-lama-inpainting
   ```
2. Install dependencies:
   ```bash
   pip install torch torchvision numpy opencv-python ruamel.yaml
   ```
3. Download **SAM & LaMa models**:
   ```bash
   # Download SAM model
   wget https://segment-anything.com/models/sam_vit_h_4b8939.pth
   
   # Clone LaMa repository & download model
   git clone https://github.com/saic-mdal/lama.git
   cd lama
   pip install -e .
   wget -P pretrained_models/big-lama https://huggingface.co/saic-mdal/lama/resolve/main/big-lama.pt
   ```

## Loading SAM Model
```python
import torch
from segment_anything import sam_model_registry, SamPredictor

# Load SAM checkpoint
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
predictor = SamPredictor(sam)
print("SAM model loaded successfully!")
```

## Loading LaMa Model
```python
import yaml
from lama_inpaint import build_lama_model

# Load LaMa configuration
lama_config = "./lama/configs/prediction/default.yaml"
lama_ckpt = "pretrained_models/big-lama/big-lama.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
with open(lama_config, "r") as f:
    config = yaml.safe_load(f)
lama_model = build_lama_model(config, lama_ckpt, device)
print("LaMa model loaded successfully!")
```

## Running Image Inpainting
```python
from lama_inpaint import inpaint_img_with_lama
import cv2

# Load image & mask
image = cv2.imread("input.jpg")
mask = cv2.imread("mask.jpg", 0)  # Grayscale mask

# Perform inpainting
inpainted_image = inpaint_img_with_lama(lama_model, image, mask, lama_config, device)

# Save result
cv2.imwrite("output.jpg", inpainted_image)
print("Inpainting completed successfully!")
```

## Summary
✔ **Step 1:** Install dependencies & clone repositories  
✔ **Step 2:** Download pretrained **SAM & LaMa** models  
✔ **Step 3:** Load **SAM for segmentation**  
✔ **Step 4:** Load **LaMa for inpainting**  
✔ **Step 5:** Run **object removal & image completion**  

## Results
- **Object removal** with precise segmentation.
- **LaMa-powered inpainting** for seamless image completion.
- **High-resolution inpainted images** with natural textures.

## Contributing
Feel free to contribute by submitting issues or pull requests!

## License
This project is licensed under the **Apache 2.0 License**.

## Acknowledgments
- **Meta AI** for developing SAM.
- **SAIC Research** for LaMa inpainting.
- **PyTorch** for deep learning frameworks.

