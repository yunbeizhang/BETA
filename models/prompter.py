"""
    source1(VP): https://github.com/hjbahng/visual_prompting
    source2(AR): https://github.com/savan77/Adversarial-Reprogramming 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter 
from torchvision import transforms
import numpy as np
import os


# Normalization constants, matching your original transform
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

class Normalize(nn.Module):
    """Applies standard normalization."""
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

    def forward(self, x):
        # Move mean and std to the same device as the input tensor
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

class InverseNormalize(nn.Module):
    """Reverses the standard normalization."""
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

    def forward(self, x):
        # Move mean and std to the same device as the input tensor
        return (x * self.std.to(x.device)) + self.mean.to(x.device)


class PadPrompter(nn.Module):
    """
    Takes a 224x224 normalized image, shrinks its content to 160x160,
    and places it inside a learnable 32px border, resulting in a 224x224 image.
    """
    def __init__(self, input_size=224, pad_size=32, output_size=224, init='random'):
        super().__init__()
        self.input_size = input_size
        self.pad_size = pad_size
        self.output_size = output_size
        self.content_size = self.output_size - 2 * self.pad_size # This will be 160

        # Helpers for normalization
        self.inv_normalize = InverseNormalize(mean=MEAN, std=STD)
        self.normalize = Normalize(mean=MEAN, std=STD)

        # Learnable parameters for the frame, initialized with random values
        if init == 'zero':
            self.pad_up = nn.Parameter(torch.zeros([1, 3, self.pad_size, self.output_size]))
            self.pad_down = nn.Parameter(torch.zeros([1, 3, self.pad_size, self.output_size]))
            self.pad_left = nn.Parameter(torch.zeros([1, 3, self.content_size, self.pad_size]))
            self.pad_right = nn.Parameter(torch.zeros([1, 3, self.content_size, self.pad_size]))
        elif init == 'random':
            self.pad_up = nn.Parameter(torch.normal(mean=0.5, std=0.5, size=[1, 3, self.pad_size, self.output_size]))
            self.pad_down = nn.Parameter(torch.normal(mean=0.5, std=0.5, size=[1, 3, self.pad_size, self.output_size]))
            self.pad_left = nn.Parameter(torch.normal(mean=0.5, std=0.5, size=[1, 3, self.content_size, self.pad_size]))
            self.pad_right = nn.Parameter(torch.normal(mean=0.5, std=0.5, size=[1, 3, self.content_size, self.pad_size]))
        
        # display the number of learnable parameters
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"PadPrompter initialized with {total_params} learnable parameters.")

    def forward(self, x):
        # x is the normalized [N, 3, 224, 224] tensor from the dataloader
        
        # 1. Inverse normalize to get pixel values in [0, 1] range
        x_pixels = self.inv_normalize(x)
        
        # 2. Resize the image content from 224x224 down to 160x160
        x_resized_pixels = transforms.functional.resize(x_pixels, size=[self.content_size, self.content_size])
        
        # 3. Re-normalize the shrunken 160x160 image
        content = self.normalize(x_resized_pixels)
        
        # 4. Assemble the final image with the learnable border
        batch_size = x.shape[0]
        
        middle_section = torch.cat([
            self.pad_left.repeat(batch_size, 1, 1, 1).to(x.device),
            content,
            self.pad_right.repeat(batch_size, 1, 1, 1).to(x.device)
        ], dim=3)
        
        final_image = torch.cat([
            self.pad_up.repeat(batch_size, 1, 1, 1).to(x.device),
            middle_section,
            self.pad_down.repeat(batch_size, 1, 1, 1).to(x.device)
        ], dim=2)
        
        return final_image
    
class ProbFuser(nn.Module):
    def __init__(self, alpha=0.3, learnable=False):
        super(ProbFuser, self).__init__()
        self.learnable = learnable
        
        if learnable:
            # Learnable parameter constrained between 0 and 1
            self.alpha_logit = nn.Parameter(torch.logit(torch.tensor(alpha, dtype=torch.float32)))
        else:
            # Fixed parameter - register as buffer so it moves with the model
            self.register_buffer('alpha_fixed', torch.tensor(alpha, dtype=torch.float32))
    
    @property
    def alpha(self):
        if self.learnable:
            return torch.sigmoid(self.alpha_logit)
        else:
            return self.alpha_fixed
    
    def forward(self, prob1, prob2):
        alpha = self.alpha
        # print(f"Using alpha: {alpha.item():.4f}")
        return alpha * prob1 + (1 - alpha) * prob2

class PadVR(nn.Module):
    def __init__(self, model, pad_size, input_size=224, output_size=224):
        super(PadVR, self).__init__()
        self.model = model
        self.prompter = PadPrompter(input_size=input_size, pad_size=pad_size, output_size=output_size)

    def forward(self, x):
        x_prompted = self.prompter(x)
        return self.model(x_prompted)