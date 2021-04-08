
from torch import nn
from torchvision import models


class KochNet(nn.Module):
    def __init__(self, in_channels=3, out_features=4096, width=64, classes=10, pretrained=False):
        super().__init__()
        self.out_features = out_features
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 1*width, kernel_size=10),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(1*width, 2*width, kernel_size=7),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(2*width, 2*width, kernel_size=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(2*width, 4*width, kernel_size=4),
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Linear(4*width*6*6, out_features),
            nn.Linear(out_features, classes),
        )

    def embed(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)

        for layer in list(self.classifier._modules.values())[:1]:
            x = layer(x)

        return x

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)

        return x


class AlexNet(nn.Module):
    def __init__(self, classes=10, pretrained=False):
        super().__init__()
        alexnet = models.alexnet(pretrained=pretrained)
        self.out_features = 4096
        self.features = alexnet.features
        self.avgpool = alexnet.avgpool
        self.classifier = nn.Sequential(
            *list(alexnet.classifier._modules.values())[:-1],
            nn.Linear(self.out_features, classes),
        )

    def embed(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)

        for layer in list(self.classifier._modules.values())[:2]:
            x = layer(x)

        return x

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)

        return x


class VGG(nn.Module):
    def __init__(self, classes=10, pretrained=False):
        super().__init__()
        vgg = models.vgg11(pretrained=pretrained)
        self.out_features = 4096
        self.features = vgg.features
        self.avgpool = vgg.avgpool
        self.classifier = nn.Sequential(
            *list(vgg.classifier._modules.values())[:-1],
            nn.Linear(self.out_features, classes),
        )

    def embed(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)

        for layer in list(self.classifier._modules.values())[:1]:
            x = layer(x)

        return x

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)

        return x


class ResNet(nn.Module):
    def __init__(self, classes=10, pretrained=False):
        super().__init__()
        resnet = models.resnet50(pretrained=pretrained)
        self.out_features = 2048
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            nn.Sequential(
                resnet.layer1,
                resnet.layer2,
                resnet.layer3,
                resnet.layer4,
            )
        )
        self.avgpool = resnet.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(self.out_features, classes),
        )

    def embed(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)

        return x

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        
        return x


def n_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == '__main__':
    k = KochNet()
    a = AlexNet()
    v = VGG()
    r = ResNet()

    print('KochNet', n_params(k))
    print('AlexNet', n_params(a))
    print('VGG', n_params(v))
    print('ResNet', n_params(r))
