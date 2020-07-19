from __future__ import division, print_function
import torch
import torch.optim as optim
import torchvision
from Alignment.data import get_datasets
from Alignment.nets import initialize_model
from Alignment.train import *
import config


def learn_alignment():

    print("Learn alignment prediction...\n")

    mypath = config.paths['my_path']

    print("PyTorch Version: ", torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)

    device = torch.device("cuda:" + str(config.regress_trans['gpu_num']) if torch.cuda.is_available() else "cpu")

    # ------- Set Training parameters: -------
    load_model = config.regress_trans['load_model']
    model_name = config.regress_trans['model_name']
    batch_size = config.regress_trans['batch_size']
    num_epochs = config.regress_trans['num_epochs']
    learning_rate = config.regress_trans['learning_rate']
    val_prct = config.regress_trans['val_prct']

    # If augmentations are needed, set to True, and choose # of them per image
    augment = config.regress_trans['augment']
    aug_num_per_img = config.regress_trans['aug_num_per_img']
    aug_std = config.regress_trans['aug_std']

    class_num = config.regress_trans['class_num'] # Fixed

    # Name for the plots
    test_name = config.regress_trans['test_name']

    # model will be saved here:
    model_path = mypath + '2_learning/Alignment/models/best_model.pt'
    data_path = mypath + 'data/'
    logs_dir = mypath + '2_learning/Alignment/logs'

    print(f'Using device: {device}')

    feature_extract = config.regress_trans['feature_extract']  # Flag for feature extracting. When False, we finetune the whole model, when True we only update the reshaped layer params

    # ------- Done Setting Training parameters. -------


    # ------- Start running the network: ---------------

    # Create NN
    if load_model:
        model_ft, normalizers = initialize_model(model_name, class_num, feature_extract, use_pretrained=True, load_from_dict=model_path)
    else:
        model_ft, normalizers = initialize_model(model_name, class_num, feature_extract, use_pretrained=True)

    # Create datasets:
    train_ds, val_ds = get_datasets(data_path, val_prct, augment, normalizers, aug_num_per_img=aug_num_per_img, aug_std=aug_std)


    # Create dataloaders
    dataloaders_dict = {'train': torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True),
                        'val': torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=True)}

    # Train model:
    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Observe that all parameters are being optimized
    # optimizer_ft = optim.SGD(model_ft.parameters(), lr=learning_rate, momentum=0.9)
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=learning_rate)

    # Setup the loss function
    criterion = torch.nn.SmoothL1Loss()

    # Train and evaluate
    model_ft, hist = train_model(
        model_ft, dataloaders_dict, criterion, optimizer_ft, device, num_epochs, model_path)

    predicted_theta = predict_test(model_ft, dataloaders_dict['val'], device)


    training_summary(hist, num_epochs, logs_dir, test_name)

    plot_results(val_ds.embedded_imgs, val_ds.thetas,
                       predicted_theta, logs_dir, test_name)

    print('\nProcess is done.')
