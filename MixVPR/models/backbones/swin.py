import torch
import torch.nn as nn
import timm
import numpy as np


class Swin(nn.Module):
    def __init__(self,
                 model_name='swinv2_base_window12to16_192to256_22kft1k',
                 pretrained=True,
                 layers_to_freeze=2
                 ):
        """Class representing the Swin (V1 and V2) backbone used in the pipeline
        Swin contains 4 layers (0 to 3), where layer 2 is the heaviest in terms of # params

        Args:
            model_name (str, optional): The architecture of the Swin backbone to instanciate. Defaults to 'swinv2_base_window12to16_192to256_22kft1k'.
            pretrained (bool, optional): Whether pretrained or not. Defaults to True.
            layers_to_freeze (int, optional): The number of blocks to freeze in layers[2]  (starting from 0) . Defaults to 2.
        """
        super().__init__()
        self.model_name = model_name
        self.layers_to_freeze = layers_to_freeze        
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.model.head = None
        
        if pretrained:
            self.model.patch_embed.requires_grad_(False)
            self.model.layers[0].requires_grad_(False)
            self.model.layers[1].requires_grad_(False)
            # layers[2] contains most of the blocks, better freeze some of them
            for i in range(layers_to_freeze*5): # we make 5 steps (swin contains lots of layers)
                self.model.layers[2].blocks[i].requires_grad_(False)
            

        if 'base' in model_name:
            out_channels = 1024
        elif 'large' in model_name:
            out_channels = 1536
        else:
            out_channels = 768
        self.out_channels = out_channels
        
        if '384' in model_name:
            self.depth = 144
        else:
            self.depth = 49
            
    def forward(self, x):
        x = self.model.forward_features(x)
        # the following is a hack to make the output of the transformer
        # as a 3D feature maps
        bs, f, c = x.shape
        x = x.view(bs, int(np.sqrt(f)), int(np.sqrt(f)), c)
        return x.permute(0,3,1,2)
    

def print_nb_params(m):
    model_parameters = filter(lambda p: p.requires_grad, m.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Trainable parameters: {params/1e6:.3}M')

if __name__ == '__main__':
    x = torch.randn(4,3,256,256)
    m = Swin(model_name='swinv2_base_window12to16_192to256_22kft1k',
                 pretrained=True,
                 layers_to_freeze=2,)
    r = m(x)
    print_nb_params(m)
    print(f'Input shape is {x.shape}')
    print(f'Output shape is {r.shape}')