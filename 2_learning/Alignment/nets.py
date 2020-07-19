import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import config

# Helper function to 'freeze' the (already trained) first layers
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True, load_from_dict=None):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0
    normalizer = None

    print(f'Initializing Model, based on: {model_name}.\n')

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.Linear(256, num_classes)
        )
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.classifier = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1)),
            nn.Conv2d(128, num_classes, kernel_size=(1, 1), stride=(1, 1)),
            nn.AdaptiveAvgPool2d(output_size=(1, num_classes)),
            nn.Flatten(),
            nn.Linear(36, 6)
        )
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "googlenet":
        """ GoogleNet
        """
        model_ft = models.googlenet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.Linear(256, num_classes)
        )

        input_size = 299

    normalizer = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}

    if load_from_dict is not None:
        dev = 'cuda:' + str(config.bg_tool['gpu_num']) if torch.cuda.is_available() else 'cpu'
        model_ft.load_state_dict(torch.load(load_from_dict, map_location=dev))
        print('Succussfully loaded pre-trained weights')

    return model_ft, normalizer
