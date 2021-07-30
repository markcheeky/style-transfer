import IPython

import numpy as np
import torchvision
import PIL.Image as pillow

from model import StyleTransfer
from preprocess import img2tensor, tensor2img

base = torchvision.models.vgg19(pretrained=True).features.eval()

content_img = pillow.open("../img/steve_jobs.jpg")
style_img = pillow.open("../img/van_gogh.jpg")

size = np.minimum(content_img.size, style_img.size) // 2
content = img2tensor(content_img, size)
style = img2tensor(style_img, size)

transfer = StyleTransfer()
paintings = transfer(base,
                     content,
                     style,
                     content_layers={4},
                     style_layers={1, 2, 3, 4, 5})

IPython.display.display(tensor2img(style))
IPython.display.display(tensor2img(content))

for i in range(1000):
    x = next(paintings).detach()
    if i % 10 == 0:
        IPython.display.display(tensor2img(x))


