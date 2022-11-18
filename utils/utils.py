import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torch
from tqdm import tqdm


def plot_2_sidebyside(img, mask):
    fig = plt.figure()
    ax1 = fig.add_subplot(121)  # left side
    ax2 = fig.add_subplot(122)  # right side
    plt.gray()
    ax1.imshow(img)
    ax2.imshow(mask)

    ax1.set_title("img")
    ax2.set_title("mask")

    plt.show()

def plot_3_sidebyside(img, img2, img3):
    fig = plt.figure()
    ax1 = fig.add_subplot(131)  # left side
    ax2 = fig.add_subplot(132)  # right side
    ax3 = fig.add_subplot(133)
    plt.gray()
    ax1.imshow(img)
    ax2.imshow(img2)
    ax3.imshow(img3)

    ax1.set_title("img")
    ax2.set_title("mask")
    ax3.set_title("pred mask")
    plt.show()


def save_model(model, epoch, model_name, optimizer, scheduler, grad_scaler, batch_size):
    torch.save({  
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict' : scheduler.state_dict(),
                        'grad_scaler': grad_scaler.state_dict(),
                        'batch_size': batch_size,
                        'lr': optimizer.param_groups[0]['lr'],
            }, model_name + '_epoch_' + str(epoch+1))

      
