from models.vgg_19 import *
import argparse
import torch
import torch.optim as optim
import os
import numpy as np
import time
import json

IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
IMAGENET_STD_NEUTRAL = [1, 1, 1]


def get_content_loss(content_image_content, generated_image_content):
    """
    The mean squared difference between the activations of the generated image and activations of the content
    image for a chosen VGG layer is computed to give the content loss.
    """
    content_loss=nn.MSELoss()(content_image_content, generated_image_content)
    return content_loss

def get_style_loss(style_image_style, generated_image_style, Ns, Ms):
    """
    On each layer included in the style representation, the element-wise mean squared difference between generated image
    layer l gram matrix and style image layer l gram matrix is computed to give the style loss.
    """

    assert len(style_image_style) == len(generated_image_style)
    style_loss = torch.zeros(1, requires_grad=True, device='cuda')

    layer_weight = np.full(len(style_image_style),1/len(style_image_style))

    layer_count=len(style_image_style)
    for i in range(layer_count):
        multiplier = ( 1 / (4 * (Ns[i]**2) * (Ms[i]**2)) )
        layer_style_loss = multiplier * nn.MSELoss()(style_image_style[i], generated_image_style[i])
        style_loss = style_loss + (layer_weight[i] * layer_style_loss)

    return style_loss


def make_loss(content_weight, style_weight, content_image_content, style_image_style, content_loss_fn, style_loss_fn, Ns, Ms):
    """
    Creates our loss function
    """

    def loss(generated_image_content, generated_image_style):
        """
        The total loss is a linear combination between the content and the style loss. Its derivative with respect to
        the pixel values can be computed using error back-propagation. This gradient is used to iteratively update the
        image until it simultaneously matches the style features of the style image and the content features of the
        content image
        """

        content_loss= content_loss_fn(content_image_content, generated_image_content )
        style_loss= style_loss_fn(style_image_style, generated_image_style,Ns, Ms)

        weighted_content_loss = content_weight * content_loss
        weighted_style_loss = style_weight * style_loss
        total_loss= weighted_content_loss + weighted_style_loss

        with torch.no_grad():
            print("Loss: {:.4f}, \t Weighted Content Loss: {:.6f}, \t Weighted Style Loss: {:.6f}, \t Content Loss: {:.6f}, \t Style Loss: {:.6f}"
                  .format(total_loss.item(), weighted_content_loss.item(), weighted_style_loss.item(), content_loss.item(), style_loss.item()))

        return total_loss

    return loss


def make_optimizer_step(model, loss_fn, optimizer, content_layer_name, style_layer_names):

    def optimizer_step(generated_image):

        generated_image_content=model.get_content(generated_image, content_layer_name, detach=False)
        generated_image_style=model.get_style(generated_image, style_layer_names, detach=False)

        loss = loss_fn(generated_image_content, generated_image_style)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        return loss

    return optimizer_step


def style_transfer(args):
    """
    Style transfer algorithm.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #We used 19-layer VGG network
    model=vgg19(device)

    # Content image resized to keep aspect ratio
    content_image=utils.preprocess_image(args.content_image, device, args.image_height)
    utils.save_img('output/content.jpg', content_image)

    # To extract image information on comparable scales, we always resized the style image to the same size as
    # the content image before computing its feature representations
    style_image=utils.preprocess_image(args.style_image, device, args.image_height, content_image.shape[-1])
    utils.save_img('output/style.jpg',style_image)

    # Generated image should be same shape as content image
    generated_image = torch.randn(content_image.shape, device=device,  requires_grad=True)

    # The content image is passed through the network and the content representation in one layer is stored.
    # The style image is passed through the network and its style representation on all layers included are computed
    # and stored.
    content_image_content = model.get_content(content_image, args.content_layer_name).detach()
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


    # Optimization loop
    if args.optimizer == 'adam':

        optimizer=optim.Adam(params=[generated_image], lr=args.lr)
        optimizer_step=make_optimizer_step(model,
                                           loss_fn,
                                           optimizer,
                                           args.content_layer_name, args.style_layer_names)
        losses=[]
        for cnt in range(args.iter_count):
            print('Iter: {} \t '.format(cnt), end='')
            loss =optimizer_step(generated_image)
            losses.append(loss)


    elif args.optimizer == 'lbfgs':

        optimizer = optim.LBFGS([generated_image], max_iter=1000, line_search_fn='strong_wolfe')

        def closure():
            if torch.is_grad_enabled():
                optimizer.zero_grad()
            generated_image_content = model.get_content(generated_image, args.content_layer_name, detach=False)
            generated_image_style = model.get_style(generated_image, args.style_layer_names, detach=False)
            loss = loss_fn(generated_image_content, generated_image_style)
            optimizer.zero_grad()
            loss.backward()
            return loss, content_loss, style_loss

        losses=[]
        for cnt in range(args.iter_count):
            print('Iter: {} \t '.format(cnt), end='') #fixme
            loss, content_loss, style_loss = optimizer.step(closure)
            losses.append(loss)



    # Storing the generated image
    timestr = time.strftime("%Y%m%d-%H%M%S")
    output_image_path = os.path.join('output', timestr + '.jpg')
    output_log_path = os.path.join('output', timestr + '.log')
    utils.save_img(output_image_path, generated_image)

    # Storing settings used to generate the image
    with open(output_log_path,'w') as f:
        json.dump(vars(args), f , indent=4)



if __name__=="__main__":

    parser=argparse.ArgumentParser()

    parser.add_argument("--content_image", type=str, default="data/content_images/tuebingen_neckarfront.jpg")
    parser.add_argument("--style_image", type=str, default="data/style_images/starry_night.jpg")
    parser.add_argument("--image_height", type=int, default=200)

    parser.add_argument("--content_weight", type=float, default=1e1)
    parser.add_argument("--style_weight", type=float, default=1e0)

    parser.add_argument("--content_layer_name", type=str, default="conv4_2")
    parser.add_argument("--style_layer_names", type=list, default=["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_3"])

    parser.add_argument("--lr", type=float, default=1e1)
    parser.add_argument("--iter_count", type=bool, default=2500)
    parser.add_argument("--optimizer", type=str, choices=['adam','lbfgs'], default='adam')

    args = parser.parse_args()

    style_transfer(args)