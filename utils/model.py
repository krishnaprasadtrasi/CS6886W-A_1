import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class VGG(nn.Module):
    def __init__(self, features, num_classes=10):
        super(VGG, self).__init__()
        self.features = features
        
        # Get the number of output channels from the last Conv2d layer
        classifier_input_size = None
        for module in reversed(list(self.features.modules())):
            if isinstance(module, nn.Conv2d):
                classifier_input_size = module.out_channels
                break
        
        if classifier_input_size is None:
            raise ValueError("No Conv2d layer found in features")
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_size, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))  # fixed output size
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()