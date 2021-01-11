import torch
import torch.nn as nn
from src import utils as utils
from torchvision import models

#Reflects the naming convention used in the paper
layer_name_to_id = {"conv1_1" : 0,
                    "relu1_1" : 1,
                    "conv1_2" : 2,
                    "relu1_2" : 3,
                    "max_pooling1" : 4,
                    "conv2_1" : 5,
                    "relu2_1" : 6,
                    "conv2_2" : 7,
                    "relu2_2" : 8,
                    "max_pooling2" : 9,
                    "conv3_1" : 10,
                    "relu3_1" : 11,
                    "conv3_2" : 12,
                    "relu3_2" : 13,
                    "conv3_3" : 14,
                    "relu3_3" : 15,
                    "conv3_4" : 16,
                    "relu3_4" : 17,
                    "max_pooling3" : 18,
                    "conv4_1" : 19,
                    "relu4_1" : 20,
                    "conv4_2" : 21,
                    "relu4_2" : 22,
                    "conv4_3" : 23,
                    "relu4_3" : 24,
                    "conv4_4" : 25,
                    "relu4_4" : 26,
                    "max_pooling4" : 27,
                    "conv5_1" : 28,
                    "relu5_1" : 29,
                    "conv5_2" : 30,
                    "relu5_2" : 31,
                    "conv5_3" : 32,
                    "relu5_3" : 33,
                    "conv5_4" : 34,
                    "relu5_4" : 35,
                    "max_pooling5" : 36}


class vgg19(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.vgg = models.vgg19(pretrained=True).features.to(device).eval()
        self.switch_to_avgpool()

        for param in self.vgg.parameters():
            param.requires_grad = False


    def switch_to_avgpool(self):
        """
        For image synthesis we found that replacing the maximum pooling operation by average pooling yields slightly
        more appealing results
        """
        for name, child in self.vgg.named_children():
            if isinstance(child, nn.MaxPool2d):
                self.vgg[int(name)] = nn.AvgPool2d(kernel_size=2, stride=2)

    def get_content(self, x, content_layer_name, detach=True):
        """
        Image's content is represented as an activation of a certain layer in the VGG network.
        """

        layer=layer_name_to_id[content_layer_name]
        activations=self.vgg[:layer](x)
        content = activations.squeeze().view(activations.shape[1],-1)

        if detach:
            content=content.detach()
            return content

        return content

    def get_style(self, x, style_layer_names, detach=True):
        """
        To obtain a representation of the style of an input image, we use a feature space designed to capture texture
        information. This feature space can be built on top of the filter responses in any layer of the network.
        It consists of the correlations between the different filter responses, where the expectation is taken over the
        spatial extent of the feature maps. These feature correlations are given by multiplying a matrix of layer
        activations with its transpose, i.e. it's a Gram matrix.
        """

        content_representations=[self.get_content(x, style_layer_name, detach) for style_layer_name in style_layer_names]
        style=[utils.get_gram_matrix(content_representations[i]) for i in range(len(content_representations))]

        return style

    def get_Ns_and_Ms(self, x, layers):
        """
        Ns represent an array of distinct filter counts in each selected layer of the VGG network used for style extraction.
        Ms represent an array of height times the width of those filters.
        These values are used in regularizing the layer's style loss in later steps and are precalculated here.
        """

        content_representations = [self.get_content(x, i, True) for i in layers]
        Ns = [content.shape[0] for content in content_representations]
        Ms = [content.shape[1] for content in content_representations]

        return Ns, Ms

    def forward(self, x):
        x_norm=self.normalize(x)
        self.vgg(x_norm)

    def normalize(self, x):
        """
        Pytorch VGG pre-trained model expects input images normalized to ImageNet standards
        """

        normalization_mean = [0.485, 0.456, 0.406]
        normalization_std = [0.229, 0.224, 0.225]

        mean = torch.tensor(normalization_mean).cuda().view(-1, 1, 1)
        std = torch.tensor(normalization_std).cuda().view(-1, 1, 1)
        return (x - mean) / std