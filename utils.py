import torch
from torchvision import transforms




def get_gram_matrix(tensor):
    """
    Returns a Gram matrix of dimension (distinct_filer_count, distinct_filter_count) where G[i,j] is the
    inner product between the vectorised feature map i and j in layer l
    """

    G = torch.mm(tensor,tensor.t())
    return G

def img_to_tensor(img, device):
    """
    Converting a PIL image to a tensor with an appropriate shape
    """
    loader = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
    ])

    img = loader(img).unsqueeze(0)
    img = img.to(device, dtype=torch.float)
    return img

def tensor_to_img(tensor):
    """
    Converting a tensor to a PIL image
    """
    unloader=transforms.ToPILImage()

    img=tensor.cpu().clone().squeeze(0)
    img=unloader(img)

    return img
