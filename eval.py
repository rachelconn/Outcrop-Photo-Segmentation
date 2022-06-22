import os
import os.path as osp

import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import DataParallel
from torch.utils.data import DataLoader

from dataset import ValDataset 
from metric import fast_hist, cal_scores
from network import EMANet 
import settings

import matplotlib.pyplot as plt
from matplotlib import cm, colors

logger = settings.logger


class Session:
    def __init__(self, dt_split):
        torch.cuda.set_device(settings.DEVICE)

        self.log_dir = settings.LOG_DIR
        self.model_dir = settings.MODEL_DIR

        self.net = EMANet(settings.N_CLASSES, settings.N_LAYERS).cuda()
        self.net = DataParallel(self.net, device_ids=[settings.DEVICE])
        dataset = ValDataset(split=dt_split)
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=False, 
                                     num_workers=2, drop_last=False)
        self.hist = 0

    def load_checkpoints(self, name):
        ckp_path = osp.join(self.model_dir, name)
        try:
            obj = torch.load(ckp_path, 
                             map_location=lambda storage, loc: storage.cuda())
            logger.info('Load checkpoint %s.' % ckp_path)
        except FileNotFoundError:
            logger.info('No checkpoint %s!' % ckp_path)
            return

        self.net.module.load_state_dict(obj['net'])

    def inf_batch(self, image, label):
        image = image.cuda()
        label = label.cuda()
        with torch.no_grad():
            logit = self.net(image)

        pred = logit.max(dim=1)[1]

        def batch_to_example(tensor):
            return np.squeeze(tensor.cpu().numpy(), axis=0)

        # Plot image
        ax1 = plt.subplot(1, 3, 1)
        ax1.set_title('Image')
        image_to_display = np.transpose(batch_to_example(image), (1, 2, 0)) / 2.64
        ax1.imshow(image_to_display)

        # Convert labels to RGB
        color_map = cm.get_cmap('hsv', settings.N_CLASSES)(np.linspace(0, 1, settings.N_CLASSES))
        color_map = np.vstack((color_map, [0., 0., 0., 1.])) # For 255 labels

        def label_to_image(labels):
            labels = batch_to_example(labels)
            labels[labels == 255] = settings.N_CLASSES
            output = color_map[labels.flatten()]
            r, c = labels.shape[:2]
            return np.reshape(output, (r, c, 4))

        label_image = label_to_image(label)
        pred_image = label_to_image(pred)

        # Plot ground truth labels
        ax2 = plt.subplot(1, 3, 2)
        ax2.set_title('Ground truth')
        ax2.imshow(label_image)

        # Plot predicted labels
        ax3 = plt.subplot(1, 3, 3)
        ax3.set_title('Predicted')
        ax3.imshow(pred_image)

        plt.show()

        self.hist += fast_hist(label, pred)


def main(ckp_name='final.pth'):
    sess = Session(dt_split='val')
    sess.load_checkpoints(ckp_name)
    dt_iter = sess.dataloader
    sess.net.eval()

    for i, [image, label] in enumerate(dt_iter):
        sess.inf_batch(image, label)
        if i % 10 == 0:
            logger.info('num-%d' % i)
            scores, cls_iu = cal_scores(sess.hist.cpu().numpy())
            for k, v in scores.items():
                logger.info('%s-%f' % (k, v))

    scores, cls_iu = cal_scores(sess.hist.cpu().numpy())
    for k, v in scores.items():
        logger.info('%s-%f' % (k, v))
    logger.info('')
    for k, v in cls_iu.items():
        logger.info('%s-%f' % (k, v))


if __name__ == '__main__':
    main()
