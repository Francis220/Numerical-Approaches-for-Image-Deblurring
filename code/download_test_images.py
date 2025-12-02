import numpy as np
from skimage import data
from PIL import Image
import os

os.makedirs('../data/original', exist_ok=True)

camera = data.camera()
Image.fromarray(camera).save('../data/original/cameraman.png')

astronaut = data.astronaut()
astronaut_gray = np.dot(astronaut[...,:3], [0.299, 0.587, 0.114])
Image.fromarray(astronaut_gray.astype(np.uint8)).save('../data/original/astronaut.png')

print("Test images downloaded successfully!")
print("Images saved in data/original/")