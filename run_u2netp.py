import os
import time
import sys
from collections import namedtuple
from pathlib import Path
import ipywidgets as widgets

import cv2
import matplotlib.pyplot as plt
import numpy as np
import openvino as ov
import torch
from IPython.display import HTML, FileLink, display


# utils_file_path = Path("notebook_utils.py")


# sys.path.append(str(utils_file_path.parent))

# from notebook_utils import load_image
def load_image(path: str) -> np.ndarray:
    import cv2
    import requests

    if path.startswith("http"):
        # Set User-Agent to Mozilla because some websites block
        # requests with User-Agent Python
        response = requests.get(path, headers={"User-Agent": "Mozilla/5.0"})
        array = np.asarray(bytearray(response.content), dtype="uint8")
        image = cv2.imdecode(array, -1)  # Loads the image as BGR
    else:
        image = cv2.imread(path)
    return image


def bg_blur(input_folder, mask_blur_radius=101, blur_padding = 5, blur_intensity = 51):
    model_ir = ov.convert_model("u2netp.onnx")
    
    # Get a list of all .jpg files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.endswith(".jpg")]
    
    output_directory = input_folder+"_bgblur"
    os.makedirs(output_directory, exist_ok=True)
    input_mean = [0,0,0]
    input_scale = [0,0,0]
    images = []
    for image_file in image_files:
        IMAGE = os.path.join(input_folder, image_file)
        image = cv2.cvtColor(src=load_image(IMAGE), code=cv2.COLOR_BGR2RGB)
        images.append(image)
        
        img_scale = [image[:,:,i].std() for i in range(3)]
        input_mean = [x + y for x, y in zip(input_mean,cv2.mean(image)[:3])]
        input_scale = [x + y for x, y in zip(input_scale,img_scale)]
        
    input_mean = [x/len(image_files) for x in input_mean]
    input_scale = [x/len(image_files) for x in input_scale]
    max_x = 0
    max_y = 0
    # Loop through each image file in the folder
    for image_file in image_files:
        # Construct the full path to the image
        IMAGE = os.path.join(input_folder, image_file)
            
        input_mean = np.array(input_mean).reshape(1, 3, 1, 1)
        input_scale = np.array(input_scale).reshape(1, 3, 1, 1)

        image = cv2.cvtColor(src=load_image(IMAGE), code=cv2.COLOR_BGR2RGB)

        resized_image = cv2.resize(src=image, dsize=(512, 512))
        # Convert the image shape to a shape and a data type expected by the network
        # for OpenVINO IR model: (1, 3, 512, 512).
        input_image = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0)

        input_image = (input_image - input_mean) / input_scale

        core = ov.Core()
        device = widgets.Dropdown(
            options=core.available_devices + ["AUTO"],
            value="AUTO",
            description="Device:",
            disabled=False,
        )

        compiled_model_ir = core.compile_model(model=model_ir, device_name=device.value)
        input_layer_ir = compiled_model_ir.input(0)
        output_layer_ir = compiled_model_ir.output(0)

        result = compiled_model_ir([input_image])[output_layer_ir]

        mask = np.rint(
            cv2.resize(src=np.squeeze(result), dsize=(image.shape[1], image.shape[0]))
        ).astype(np.uint8)

        mask = cv2.GaussianBlur(mask*255, (mask_blur_radius, mask_blur_radius), blur_padding)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        max_x = max(max(abs(x-image.shape[1]/2),abs((x+w)-image.shape[1]/2)),max_x)
        max_y = max(max(abs(y-image.shape[0]/2),abs((y+h)-image.shape[0]/2)),max_y)
        blurred_image = cv2.GaussianBlur(image, (blur_intensity, blur_intensity), 0)

        image_c = image.copy()

        image_c[mask == 0] = blurred_image[mask == 0]

        output_filename = os.path.join(output_directory, os.path.basename(IMAGE))
        image_c_rgb = cv2.cvtColor(image_c, cv2.COLOR_BGR2RGB)
        cv2.imwrite(output_filename, image_c_rgb)

        print(f"Processed: {IMAGE}")
    return output_filename, (max_x, max_y)