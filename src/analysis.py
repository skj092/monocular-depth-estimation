from glob import glob
from PIL import Image
import random

path = 'val_extracted'

images = glob('val_extracted/val/**/**/**/*.png', recursive=True)
print(f'Number of images: {len(images)}')


img = Image.open(random.choice(images))
print(f'Image size: {img.size}')
