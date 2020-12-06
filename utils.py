import torch
from PIL import Image
from torchvision import transforms
import cv2
import numpy as np

IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
IMAGENET_STD_NEUTRAL = [1, 1, 1]

def get_gram_matrix(tensor):
    """
    Returns a Gram matrix of dimension (distinct_filer_count, distinct_filter_count) where G[i,j] is the
    inner product between the vectorised feature map i and j in layer l
    """

    G = torch.mm(tensor,tensor.t())
    return G

def preprocess_image(path, device, target_height, target_width=None):
    """
    Converting the image to a tensor with the appropriate shape
    """

    img = Image.open(path)
    w,h = img.size

    #Keeps ascpect ratio if width not specified
    if target_width==None:
        target_width= int(( w / h ) * target_height)

    loader = transforms.Compose([
        transforms.PILToTensor(),
        transforms.Resize((target_height, target_width)),
        transforms.Lambda(lambda x: x.float()),
        # transforms.Lambda(lambda x: x.mul(1./255.)), # Worked better with [0, 255] range
        transforms.Normalize(mean=IMAGENET_MEAN_255, std=IMAGENET_STD_NEUTRAL)
    ])

    img = loader(img).to(device).unsqueeze(0)
    return img

def save_img(path, img):
    img = img.squeeze().to('cpu').detach().numpy()
    img = img.copy()
    img = np.moveaxis(img, 0, 2)
    img = img + np.array(IMAGENET_MEAN_255).reshape((1,1,3))
    img = np.clip(img, 0, 255).astype('uint8')
    cv2.imwrite(path,  img[:, :, ::-1])
