import sys
import torch
import config
sys.path.append('../2_learning/Alignment')
from nets import initialize_model
from data import get_datasets_for_test
import train


def run_theta_regressor(test_imgs, mypath):
    print('\nGet predictions from net ...')

    # ------- Model parameters: -------
    model_name = config.regress_trans['model_name']
    batch_size = config.regress_trans['batch_size']
    model_path = mypath + '2_learning/Alignment/models/best_model.pt'
    class_num = config.regress_trans['class_num']  # Fixed
    device = torch.device("cuda:" + str(config.bg_tool['gpu_num']) if torch.cuda.is_available() else "cpu")

    # ------- Run prediction: ----------
    model_ft, normalizers = initialize_model(model_name, class_num, False, use_pretrained=True, load_from_dict=model_path)

    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Create dataset:
    val_ds = get_datasets_for_test(test_imgs, normalizers)

    # Create dataloaders
    val_data = torch.utils.data.DataLoader(val_ds, batch_size=val_ds.imgs.shape[0])

    # Get predictions
    preds = train.predict_test(model_ft, val_data, device)

    print('preds shape: ', preds.shape)

    return preds.detach().cpu().numpy()
