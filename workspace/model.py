import torch.nn as nn

class Head(nn.Module):
    def __init__(self, in_channels: int = 32, output_class_num: int = 10) -> None:
        super(Head, self).__init__()
        self.fully_connected = nn.Linear(in_channels, output_class_num)
        self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        y = self.fully_connected(x)
        y = self.activation(y)
        return y

class SimpleCNN(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 32, class_num: int = 10) -> None:
        super(SimpleCNN, self).__init__()
        whole_layer_list = list()
        whole_layer_list.extend(self._create_cnn_block(in_channels, 32))
        whole_layer_list.extend(self._create_cnn_block(32, 64))
        whole_layer_list.extend(self._create_cnn_block(64, out_channels))
        whole_layer_list.extend([nn.AdaptiveAvgPool2d((1, 1))])
        self.whole_layer = nn.Sequential(*whole_layer_list)
        self.flat = nn.Flatten(start_dim=1)
        self.head = Head(out_channels, class_num)

    def forward(self, x):
        y = self.whole_layer(x)
        y = self.flat(y)
        y = self.head(y)
        return y
    
    def _create_cnn_block(self, in_channels: int, out_channels: int):
        layer_list = [
            nn.Conv2d(in_channels, out_channels, 3, 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        return nn.Sequential(*layer_list)
    