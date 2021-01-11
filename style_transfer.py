import torch.optim as optim

import os
import time
import json
import argparse
from PIL import Image

from models.vgg19 import *
from loss import *

def style_transfer(args):
    """
    Style transfer algorithm.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # We used a 19-layer VGG network
    model = vgg19(device)

    # Content image resized to keep aspect ratio
    content_image = utils.img_to_tensor(Image.open(args.content_image_pth), device, args.image_height, args.image_width)

    # To extract image information on comparable scales, we always resized the style image to the same size as
    # the content image before computing its feature representations
    style_image = utils.img_to_tensor(Image.open(args.style_image_pth), device, args.image_height, args.image_width)

    # Generated image should be same shape as content image
    generated_image = torch.randn(content_image.data.size(), device=device,  requires_grad=True)

    # The content image is passed through the network and the content representation in one layer is stored.
    # The style image is passed through the network and its style representation on all layers included are computed
    # and stored.
    content_image_content = model.get_content(content_image, args.content_layer_name)
    style_image_style = model.get_style(style_image, args.style_layer_names)

    # Extracting layer output dimensions for the style image. These dimensions are needed in regularizing the style loss
    # function
    Ns, Ms = model.get_Ns_and_Ms(style_image, args.style_layer_names)

    loss_fn = make_loss(args.content_weight, args.style_weight,
                        content_image_content,
                        style_image_style,
                        get_content_loss,
                        get_style_loss,
                        Ns, Ms)

    optimizer = optim.LBFGS([generated_image])
    losses=[]

    for cnt in range(args.iter_count):

        def closure():
            generated_image.data.clamp_(0,1)
            optimizer.zero_grad()

            generated_image_content = model.get_content(generated_image, args.content_layer_name, detach=False)
            generated_image_style = model.get_style(generated_image, args.style_layer_names, detach=False)

            loss = loss_fn(generated_image_content, generated_image_style)

            loss.backward()
            return loss

        print('Iter: {} \t '.format(cnt), end='') #fixme
        loss= optimizer.step(closure)
        losses.append(loss)

    generated_image.data.clamp_(0, 1)


    # Storing the generated image
    timestr = time.strftime("%Y%m%d-%H%M%S")
    output_image_path = os.path.join('output', timestr + '.jpg')
    output_log_path = os.path.join('output', timestr + '.log')

    utils.tensor_to_img(generated_image).save(output_image_path)

    # Storing settings used to generate the image
    with open(output_log_path,'w') as f:
        json.dump(vars(args), f, indent=4)



if __name__=="__main__":

    parser=argparse.ArgumentParser()

    parser.add_argument("--content_image_pth", type=str, default="data/content_images/tuebingen_neckarfront.jpg")
    parser.add_argument("--style_image_pth", type=str, default="data/style_images/gogh2.jpg")
    parser.add_argument("--image_height", type=int, default=512)
    parser.add_argument("--image_width", type=int, required=False, default=None) # Aspect ratio kept if not specified

    parser.add_argument("--content_weight", type=float, default=1)
    parser.add_argument("--style_weight", type=float, default=1e7)

    parser.add_argument("--content_layer_name", type=str, default="conv3_2")
    parser.add_argument("--style_layer_names", type=list, default=["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"])

    parser.add_argument("--iter_count", type=bool, default=30)
    parser.add_argument("--optimizer", type=str, choices=['lbfgs'], default='lbfgs')

    args = parser.parse_args()

    style_transfer(args)
