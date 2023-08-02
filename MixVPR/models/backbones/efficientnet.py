import torch
import torch.nn as nn
import timm
import numpy as np

class EfficientNet(nn.Module):
    def __init__(self,
                 model_name='efficientnet_b0',
                 pretrained=True,
                 layers_to_freeze=4,
                 ):
        """Class representing the EfficientNet backbone used in the pipeline
        EfficientNet contains 7 efficient blocks (0 to 6),
        we don't take into account the global pooling and the last fc

        Args:
            model_name (str, optional): The architecture of the efficietnet backbone to instanciate. Defaults to 'efficientnet_b0'.
            pretrained (bool, optional): Whether pretrained or not. Defaults to True.
            layers_to_freeze (int, optional): The number of blocks to freeze (starting from 0) . Defaults to 4.
        """
        super().__init__()
        self.model_name = model_name
        self.layers_to_freeze = layers_to_freeze
        self.model = timm.create_model(model_name=model_name, pretrained=pretrained)
        
        # freeze only if the model is pretrained
        if pretrained:
            if layers_to_freeze >= 0:
                self.model.conv_stem.requires_grad_(False)
                self.model.blocks[0].requires_grad_(False)
                self.model.blocks[1].requires_grad_(False)
            if layers_to_freeze >= 1:
                self.model.blocks[2].requires_grad_(False)
            if layers_to_freeze >= 2:
                self.model.blocks[3].requires_grad_(False)
            if layers_to_freeze >= 3:
                self.model.blocks[4].requires_grad_(False)
            if layers_to_freeze >= 4:
                self.model.blocks[5].requires_grad_(False)

        self.model.global_pool = None
        self.model.fc = None
        
        out_channels = 1280 # for b0 and b1
        if 'b2' in model_name:
            out_channels = 1408
        elif 'b3' in model_name:
            out_channels = 1536
        elif 'b4' in model_name:
            out_channels = 1792
        self.out_channels = out_channels
        
    def forward(self, x):
        x = self.model.forward_features(x)
        return x


def print_nb_params(m):
    model_parameters = filter(lambda p: p.requires_grad, m.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Trainable parameters: {params/1e6:.3}M')


if __name__ == '__main__':
    x = torch.randn(4, 3, 320, 320)
    m = EfficientNet(model_name='efficientnet_b0',
                  pretrained=True,
                  layers_to_freeze=0,
                )
    r = m(x)
    print_nb_params(m)
    print(f'Input shape is {x.shape}')
    print(f'Output shape is {r.shape}')
