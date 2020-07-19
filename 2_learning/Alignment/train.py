from __future__ import division, print_function
import copy
import time
import cv2
import numpy as np
import torch
from scipy.linalg import expm, logm
from utils.image_warping import warp_image
from utils.Plots import *


def train_model(model, dataloaders, criterion, optimizer, device, num_epochs, model_path, is_inception=False):
    since = time.time()

    train_loss_history = []
    val_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.inf

    for epoch in range(num_epochs):
        print('\nEpoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 20)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.float)
                # print(f'input shape: {inputs.shape}')
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    preds = model(inputs)
                    loss = criterion(preds, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                # running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            # running_corrects.double()/len(dataloaders[phase].dataset)
            epoch_acc = 0

            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_loss_history.append(epoch_loss)
            if phase == 'train':
                train_loss_history.append(epoch_loss)

        # load best model weights and save it
        model.load_state_dict(best_model_wts)
        torch.save(model.state_dict(),model_path)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))


    # plot_results
    return model, [train_loss_history, val_loss_history]


def predict_test(model, dataloader, device):
    for inputs, labels in dataloader:
        inputs = inputs.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.float)
        preds = model(inputs)

    return preds


def training_summary(hist, num_epochs, logs_dir, test_name):
    train, val = hist
    plt.title("Loss vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Loss")
    plt.plot(range(1, num_epochs+1), train, label="train loss")
    plt.plot(range(1, num_epochs+1), val, label="val loss")
    # plt.xticks(np.arange(1, num_epochs+1, 1.0))
    plt.legend()
    plt.savefig(f'{logs_dir}/loss_{test_name}.png')


def plot_results(test_x_embd, test_theta, predicted_theta, logs_dir, test_name):
    print('-' * 20)
    print('Plotting Panorama results:')

    img_emb_sz = (test_x_embd[0].shape[0] - 400, test_x_embd[0].shape[1] - 400)
    predicted_theta = predicted_theta.cpu().detach().numpy()
    imgs_trans_lrn = []
    imgs_trans_gt = []
    for i in range(len(test_x_embd)):
        I = test_x_embd[i, ...]

        T_lrn = convert_to_expm(predicted_theta[i, ...])
        T_lrn = np.reshape(T_lrn, (2, 3))
        T_gt = convert_to_expm(test_theta[i, ...])
        T_gt = np.reshape(T_gt, (2, 3))

        I_T_lrn, _ = warp_image(I, T_lrn, cv2.INTER_CUBIC)
        I_T_lrn = np.abs(I_T_lrn/np.nanmax(I_T_lrn))
        I_T_lrn = embed_to_normal_sz_image(I_T_lrn, img_emb_sz)
        imgs_trans_lrn.append(I_T_lrn)

        I_T_gt, _ = warp_image(I, T_gt, cv2.INTER_CUBIC)
        I_T_gt = np.abs(I_T_gt/np.nanmax(I_T_gt))
        I_T_gt = embed_to_normal_sz_image(I_T_gt, img_emb_sz)
        imgs_trans_gt.append(I_T_gt)

    # --------- build panoramic images of learned theta:------------
    panoramic_img_lrn = np.nanmedian(imgs_trans_lrn, axis=0)  # nanmean
    fig1 = open_figure(1, f'Panoramic Image Predicted - {test_name}', (3, 2))
    PlotImages(1, 1, 1, 1, [panoramic_img_lrn], [''],
               'gray', axis=False, colorbar=False)

    # --------- build panoramic images of ground-truth theta:--------
    panoramic_img_gt = np.nanmedian(imgs_trans_gt, axis=0)  # nanmean
    fig2 = open_figure(2, f'Panoramic Image GT - {test_name}', (3, 2))
    PlotImages(2, 1, 1, 1, [panoramic_img_gt], [''],
               'gray', axis=False, colorbar=False)

    fig1.savefig(f'{logs_dir}/Pano_{test_name}_pred.png', dpi=1000)
    fig2.savefig(f'{logs_dir}//Pano_{test_name}_gt.png', dpi=1000)
    plt.show()
    print('-' * 20)
    print('Done Plotting.')


def plot_augmentations(images, plot_name, img_num=9):
    print('-' * 20)
    print('Plotting Augmentations:')

    random_imgs = images[np.random.choice(
        range(images.shape[0]), size=img_num)]

    cols = int(np.sqrt(img_num))
    n_images = len(random_imgs)
    titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(random_imgs, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.suptitle(plot_name)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()
    print('-' * 20)
    print('Done Plotting.')
    return fig


# Get se transformation, in shape: (1,6)
# Return SE transformation, in shape (1,6)
def convert_to_expm(T):
    T = np.reshape(T, (2, 3))
    bottom = np.zeros((1, 3))
    T = np.concatenate((T, bottom), axis=0)
    T_exmp = expm(T)[0:2, :]
    return T_exmp.ravel()


def embed_to_normal_sz_image(img, img_emb_sz):
    img_big_emb_sz = img.shape
    st_y = (img_big_emb_sz[0] - img_emb_sz[0]) // 2
    st_x = (img_big_emb_sz[1] - img_emb_sz[1]) // 2
    return img[st_y:st_y + img_emb_sz[0], st_x:st_x + img_emb_sz[1], :]


def embed_to_big_image(img, img_big_emb_sz, img_emb_sz, img_sz):
    I = np.zeros((img_big_emb_sz[0], img_big_emb_sz[1], 3))
    I[::] = np.nan
    start_idx_y = ((img_big_emb_sz[0] - img_emb_sz[0]) //
                   2) + (img_emb_sz[0] - img_sz[0]) // 2
    start_idx_x = ((img_big_emb_sz[1] - img_emb_sz[1]) //
                   2) + (img_emb_sz[1] - img_sz[1]) // 2
    I[start_idx_y:start_idx_y + img_sz[0],
        start_idx_x:start_idx_x + img_sz[1], :] = img
    return np.abs(I / np.nanmax(I))
