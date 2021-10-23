from torchvision import transforms
import numpy as np
from PIL import Image
class toBinary(object):
    def __call__(self, label):
        label = np.array(label)
        # print(image)
        label = label * (label > 127)
        label = Image.fromarray(label)
        return label
