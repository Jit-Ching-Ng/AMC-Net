import os
import shutil
from evaluate.metric import *
from dataset.utils import get_transform
import cv2
import torch
from network.AMC_Net import ACM_Net
from tqdm import tqdm
from config import BaseConfig


def get_ACM_model(config):
    net = ACM_Net(config.in_channels, config.out_channels, mid_channels=config.mid_channels,
                  norm=config.norm, interpolation_mode=config.interpolation_mode).to(config.device)
    net.load_state_dict(torch.load(model_path, map_location=config.device))
    return net


def get_metric_value(predict_mask, mask):
    mask_dice, mask_iou, mask_recall, mask_precision, mask_jaccard_index = (
        dice_coefficient(predict_mask, mask),
        iou(predict_mask, mask),
        recall(predict_mask, mask),
        precision(predict_mask, mask),
        jaccard_index(predict_mask, mask))
    return mask_dice, mask_iou, mask_recall, mask_precision, mask_jaccard_index


def ACM_predict(net, image, mask, threshold):
    mean, std = image.mean(axis=(0, 1), keepdims=True), image.std(axis=(0, 1), keepdims=True)
    image = (image - mean) / std
    image = cv2.resize(image, (config.image_size, config.image_size), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (config.image_size, config.image_size), interpolation=cv2.INTER_LINEAR)
    mask = mask > 128
    image = torch.as_tensor(np.float32(image).transpose(2, 0, 1)).unsqueeze(0).to(config.device)
    predict, rec_image = net(image)
    predict = predict.detach().cpu().numpy()[0][0]
    rec_image = rec_image.detach().cpu().numpy()
    predict_mask = predict > threshold
    rec_image = np.transpose(rec_image[0], (1, 2, 0)) * std + mean
    mask_dice, mask_iou, mask_recall, mask_precision, mask_jaccard_index = get_metric_value(predict_mask, mask)
    return rec_image, predict_mask, predict, mask_dice, mask_iou, mask_recall, mask_precision, mask_jaccard_index


if __name__ == '__main__':
    test_dataset_path = '../data/SegDataset/TestDataset'
    test_dataset_name_list = ["CVC-300", "CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB", "Kvasir"]
    config = BaseConfig()
    threshold = 0.5
    result_dir = os.path.join(config.result_dir, config.experiment_name, "Test")
    transform = get_transform(config, is_train=False)
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.mkdir(result_dir)

    for dataset_name in test_dataset_name_list:
        dataset_result_dir = os.path.join(result_dir, dataset_name)
        if not os.path.exists(dataset_result_dir):
            os.mkdir(dataset_result_dir)
    model_path = os.path.join(config.result_dir, config.experiment_name, "Train", f"epoch_{config.n_epochs-1}.pth")
    net = get_ACM_model(config)

    for index, dataset_name in enumerate(test_dataset_name_list):
        dataset_result_dir = os.path.join(result_dir, dataset_name)
        image_dir_path = os.path.join(test_dataset_path, dataset_name, "images")
        mask_dir_path = os.path.join(test_dataset_path, dataset_name, "masks")
        avg_dice = 0
        avg_iou = 0
        avg_recall = 0
        avg_precision = 0
        avg_jaccard_index = 0
        for image in tqdm(os.listdir(image_dir_path)):
            image_path = os.path.join(image_dir_path, image)
            mask_path = os.path.join(mask_dir_path, image)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            (rec_image, predict_mask, predict, mask_dice,
             mask_iou, mask_recall, mask_precision, mask_jaccard_index) = ACM_predict(net, image, mask, threshold)
            save_image_path = os.path.join(dataset_result_dir, "rec_" + os.path.basename(image_path))
            save_mask_path = os.path.join(dataset_result_dir, "mask_" + os.path.basename(image_path))
            save_predict_path = os.path.join(dataset_result_dir, "predict_" + os.path.basename(image_path))
            cv2.imwrite(save_image_path, rec_image[:,:,[2,1,0]])
            cv2.imwrite(save_mask_path, predict_mask * 255)
            cv2.imwrite(save_predict_path, predict)
            avg_dice += mask_dice
            avg_iou += mask_iou
            avg_recall += mask_recall
            avg_precision += mask_precision
            avg_jaccard_index += mask_jaccard_index
        avg_dice /= len(os.listdir(image_dir_path))
        avg_iou /= len(os.listdir(image_dir_path))
        avg_recall /= len(os.listdir(image_dir_path))
        avg_precision /= len(os.listdir(image_dir_path))
        avg_jaccard_index /= len(os.listdir(image_dir_path))

        print(f"---------------For Dataset {dataset_name}---------------")
        print(f"Dice: {avg_dice}")
        print(f"IOU: {avg_iou}")
        print(f"Recall: {avg_recall}")
        print(f"Precision: {avg_precision}")
        print(f"Jaccard: {avg_jaccard_index}\n")
