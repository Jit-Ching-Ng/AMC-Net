import os.path
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset.segmentation_dataset import SegmentationDataset
from config import BaseConfig
from utils import prepare_work, update_learning_rate
from network.utils import get_model, LambdaLR
from evaluate.metric import MaskLoss, dice_coefficient


def train(config):
    phase_dir = prepare_work(config)
    model = get_model(config)
    train_loader = DataLoader(dataset=SegmentationDataset(config, config.train_path, is_train=True),
                              batch_size=config.batch_size, shuffle=True, num_workers=2)
    optimizer = torch.optim.AdamW(model.parameters(), config.lr, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                     lr_lambda=LambdaLR(config.n_epochs, config.decay_epoch).step)
    mask_loss_function = MaskLoss()
    identity_loss_function = nn.L1Loss()

    mask_loss_list = []
    identity_loss_list = []
    total_loss_list = []

    for epoch in range(config.n_epochs):
        model.train()
        if epoch != 0:
            update_learning_rate(optimizer, lr_scheduler)

        epoch_mask_loss = 0
        epoch_identity_loss = 0
        epoch_total_loss = 0
        for image, mask in tqdm(train_loader):
            optimizer.zero_grad()
            image, mask = image.to(config.device), mask.to(config.device).unsqueeze(1)
            mask_prediction, rec_image = model(image)
            mask_loss = mask_loss_function(mask_prediction, mask)
            identity_loss = identity_loss_function(rec_image, image)
            total_loss = mask_loss + identity_loss * config.lambda_identity
            total_loss.backward()
            optimizer.step()
            epoch_mask_loss += mask_loss.item()
            epoch_identity_loss += identity_loss.item()
            epoch_total_loss += total_loss.item()
        epoch_mask_loss /= len(train_loader)
        epoch_identity_loss /= len(train_loader)
        epoch_total_loss /= len(train_loader)
        print('Epoch={:03d}/{:03d}, mask_loss={:0.4f}, identity_loss={:0.4f} , total_loss={:0.4f}'.format(
            epoch + 1, config.n_epochs, epoch_mask_loss, epoch_identity_loss, epoch_total_loss))
        mask_loss_list.append(epoch_mask_loss)
        identity_loss_list.append(epoch_identity_loss)
        total_loss_list.append(epoch_total_loss)

        if (epoch + 1) % config.save_frequency == 0:
            save_path = os.path.join(phase_dir, f"epoch_{epoch}.pth")
            torch.save(model.cpu().state_dict(), save_path)
            model.to(config.device)

    print("\nTraining Finish!\n")

    figure, ax = plt.subplots(figsize=(9.8, 9.8))
    ax.plot(range(config.n_epochs), mask_loss_list, label='Mask Loss', color='green')
    if config.lambda_identity != 0:
        ax.plot(range(config.n_epochs), identity_loss_list, label='Identity Loss', color='#005aff')
        ax.plot(range(config.n_epochs), total_loss_list, label='Total Loss', color='black')
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.set_xlabel('Epoch', fontdict={'fontsize': 18})
    ax.set_ylabel('Loss', fontdict={'fontsize': 18})
    ax.set_title(f"Middle={config.mid_channels}", fontdict={'fontsize': 18})
    if config.lambda_identity != 0:
        ax.legend(loc='upper right', fontsize=20)
    plt.tight_layout()
    plt.savefig(f"Middle Channel={config.mid_channels} Lambda={config.lambda_identity}.png")


if __name__ == '__main__':
    base_config = BaseConfig()
    train(base_config)
