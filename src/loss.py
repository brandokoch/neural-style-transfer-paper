import torch
import numpy as np
import torch.nn.functional as F


def get_content_loss(content_image_content, generated_image_content):
    """
    The mean squared difference between the activations of the generated image and activations of the content
    image for a chosen VGG layer is computed to give the content loss.
    """
    content_loss=F.mse_loss(content_image_content, generated_image_content)
    return content_loss


def get_style_loss(style_image_style, generated_image_style, Ns, Ms):
    """
    On each layer included in the style representation, the element-wise mean squared difference between generated image
    layer l gram matrix and style image layer l gram matrix is computed to give the style loss.
    """

    assert len(style_image_style) == len(generated_image_style)
    style_loss = torch.zeros(1, device='cuda')

    layer_weight = np.full(len(style_image_style),1/len(style_image_style))

    layer_count = len(style_image_style)
    for i in range(layer_count):
        multiplier = ( 1 / (4 * (Ns[i]**2) * (Ms[i]**2)) )
        layer_style_loss = multiplier * F.mse_loss(style_image_style[i], generated_image_style[i])
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

        content_loss = content_loss_fn(content_image_content, generated_image_content )
        style_loss = style_loss_fn(style_image_style, generated_image_style,Ns, Ms)

        weighted_content_loss = content_weight * content_loss
        weighted_style_loss = style_weight * style_loss
        total_loss = weighted_content_loss + weighted_style_loss

        with torch.no_grad():
            print("Loss: {:.4f}, \t "
                  "Weighted Content Loss: {:.6f}, \t "
                  "Weighted Style Loss: {:.6f}, \t "
                  "Content Loss: {:.6f}, \t "
                  "Style Loss: {:.6f}"
                  .format(total_loss.item(), weighted_content_loss.item(), weighted_style_loss.item(), content_loss.item(), style_loss.item()))

        return total_loss

    return loss