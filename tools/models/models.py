import torchvision
import torch
import torch.nn.functional as F


def mobilenetv2(pretrained=True):
    model = torchvision.models.mobilenet_v2(pretrained=pretrained)
    return model

def densenet(pretrained=True):
    model = torchvision.models.densenet161(pretrained=pretrained)
    return model


class TransferredNetworkDenseNet(torch.nn.Module):
    def __init__(self, pretrained_model, num_classes):
        super(TransferredNetworkDenseNet, self).__init__()
        self.pretrained_model = pretrained_model
        self.fc = torch.nn.Linear(in_features=pretrained_model.classifier.in_features, out_features=num_classes)
        
    
    def forward(self, x):
        x = self.pretrained_model.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        
        x = self.fc(x)
        return x
    
class TransferredNetworkMobileNet(torch.nn.Module):
    def __init__(self, pretrained_model, num_classes):
        super(TransferredNetworkMobileNet, self).__init__()
        
        self.pretrained_model = pretrained_model
        self.fc = torch.nn.Linear(in_features=pretrained_model.classifier[1].in_features, out_features=num_classes)
        
    def forward(self, x):
        x = self.pretrained_model.features(x)
        pooled_features = x.mean([2, 3])
        
        x = self.fc(pooled_features)
        return x  
    

def make(model_name, num_classes):

    if model_name == 'densenet':
        model = densenet(pretrained=True)
        model = TransferredNetworkDenseNet(pretrained_model=model,
                                           num_classes=num_classes)
        return model
    elif model_name == 'mobilenet':
        model = mobilenetv2(pretrained=True)
        model = TransferredNetworkMobileNet(pretrained_model=model,
                                            num_classes=num_classes)
        return model


def test_model_outputs():
    model = make('densenet', 2)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)

    model = make('mobilenet', 2)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)
    
test_model_outputs()