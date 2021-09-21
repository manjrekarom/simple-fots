from utils import move_to
from loss.fotsloss import FOTSLoss
import time
import json
import pathlib

import torch
from torch.optim import SGD
from torch.nn import functional as F

from model.fots import FOTS
from data.textdataset import ICDAR
from model.textboxes import TextBoxes
from data.loaders import OCRDataLoaderFactory


def main(config):
    # device
    device = torch.device('cpu')
    if torch.cuda.is_available() and config['cuda']:
        device = torch.device('cuda:0')

    # dataset and dataloaders
    data_root = pathlib.Path(config['data_loader']['data_dir'])
    ICDARDataset2015 = ICDAR(data_root, year='2015')
    data_loader = OCRDataLoaderFactory(config, ICDARDataset2015)
    train_loader = data_loader.train()
    val_loader = data_loader.val()

    # model
    model_type = config['arch']['model']
    backbone = config['arch']['backbone']['model']
    pretrained = config['arch']['backbone']['pretrained']
    if model_type == 'fots':
        model = FOTS(backbone=backbone, pretrained=pretrained)
    elif model_type == 'tb':
        model = TextBoxes(pretrained=pretrained)
    else:
        model = FOTS()
    model.to(device)

    # trainer
    optimizer = SGD(model.parameters(), lr=1e-3)
    # pixel wise multinomial log loss
    criterion = FOTSLoss()

    last_val = 1e20
    best_val = 1e20
    epochs = config['trainer']['epochs']
    for ITER in range(epochs):
        # train loop
        train_images, train_loss = 0, 0.0
        start = time.time()
        for batch_id, gt in enumerate(train_loader):
            imagePaths, img, gt_score, gt_geo, gt_training_mask, gt_transcripts, gt_boxes, mapping = move_to(device, *gt)
            # out is a tuple of length 4; first item is stride 1/4 of original input
            pred_score, pred_geo = model(img)
            loss = criterion(pred_score, pred_geo, gt_score, gt_geo)
            train_loss += loss.item()
            train_images += config['data_loader']['batch_size']
            loss.backward()
            optimizer.step()
            if (batch_id + 1) % 25 == 0:
                print(f'--finished {train_images + 1} batches, train loss/image={train_loss/train_images}, (images/sec={train_images/(time.time()-start)})')
        print(f'iter {ITER}: train loss/image={train_loss/train_images}, (images/sec={train_images/(time.time()-start)})')
        
        # dev/val loop
        val_images, val_loss = 0, 0.0
        start = time.time()
        for batch_id, gt in enumerate(val_loader):
            with torch.no_grad():
                imagePaths, img, gt_score, gt_geo, gt_training_mask, gt_transcripts, gt_boxes, mapping = move_to(device, *gt)
                # out is a tuple of length 4; first item is stride 1/4 of original input
                pred_score, pred_geo = model(img)
                loss = criterion(pred_score, pred_geo, gt_score, gt_geo)
                val_loss += loss.item()
                val_images += config['data_loader']['batch_size']
        last_val = val_loss
        if best_val > last_val:
            torch.save(model, 'model.pth')
        print(f'iter {ITER}: val loss/image={val_loss/val_images}, (images/sec={val_images/(time.time()-start)})')
        # # generate a few examples
        # for (img, sgmt) in example_loader:
        #     with torch.no_grad():
        #         img = img.to(device), sgmt = sgmt.to(device)
        #         output = F.softmax(model(img))


if __name__ == "__main__":
    config = json.load(open('config.json'))
    main(config)
