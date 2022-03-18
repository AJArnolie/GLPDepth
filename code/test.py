'''
Doyeon Kim, 2022
'''

import os
import cv2
import numpy as np
from collections import OrderedDict
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

import utils.logging as logging
import utils.metrics as metrics
from models.model import GLPDepth
from dataset.base_dataset import get_dataset
from configs.test_options import TestOptions

metric_name = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log',
               'log10', 'silog']


def main():
    # experiments setting
    opt = TestOptions()
    args = opt.initialize().parse_args()
    print(args)

    if args.gpu_or_cpu == 'gpu':
        device = torch.device('cuda')
        cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    if args.save_eval_pngs or args.save_visualize:
        result_path = os.path.join(args.data_path, "GLP_depth")
        logging.check_and_make_dirs(result_path)
        print("Saving result images in to %s" % result_path)
    
    print("\n1. Define Model")
    model = GLPDepth(max_depth=args.max_depth, is_train=False).to(device)
    model_weight = torch.load(args.ckpt_dir)
    if 'module' in next(iter(model_weight.items()))[0]:
        model_weight = OrderedDict((k[7:], v) for k, v in model_weight.items())
    model.load_state_dict(model_weight)
    model.eval()

    print("\n3. Inference & Evaluate")
    path = os.path.join(args.data_path, "downsampled_images") # the dir of imgs
    imgs_list = os.listdir(path)
    transform = transforms.ToTensor()
    for idx, i in enumerate(imgs_list):
        print(os.path.join(path, i))
        with torch.no_grad():
            img = cv2.imread(os.path.join(path, i))
            img_resize = cv2.resize(img, (int(img.shape[1]), int(img.shape[0])), interpolation=cv2.INTER_LINEAR)
            img_torch = transform(img_resize).cuda()
            img_torch = img_torch[None, :, :, :].cuda()
            pred = model(img_torch)
        pred_d = pred['pred_d']
       
        if args.save_eval_pngs:
            save_path = os.path.join(result_path, i)
            if save_path.split('.')[-1] == 'jpg':
                save_path = save_path.replace('jpg', 'png')
            pred_d = pred_d.squeeze()
            pred_d = pred_d.cpu().numpy() * 800 #256.0
            cv2.imwrite(save_path, pred_d.astype(np.uint16),
                            [cv2.IMWRITE_PNG_COMPRESSION, 0])
            
        #if args.save_visualize:
        #    save_path = os.path.join(result_path, i)
        #    pred_d_numpy = pred_d.squeeze().cpu().numpy()
        #    pred_d_numpy = (pred_d_numpy / pred_d_numpy.max()) * 255
        #    pred_d_numpy = pred_d_numpy.astype(np.uint8)
        #    pred_d_color = cv2.applyColorMap(pred_d_numpy, cv2.COLORMAP_RAINBOW)
        #    cv2.imwrite(save_path, pred_d_color)
        logging.progress_bar(idx, len(imgs_list), 1, 1)

    #if args.do_evaluate:
    #    for key in result_metrics.keys():
    #        result_metrics[key] = result_metrics[key] / (batch_idx + 1)
    #    display_result = logging.display_result(result_metrics)
    #    if args.kitti_crop:
    #        print("\nCrop Method: ", args.kitti_crop)
    #    print(display_result)

    print("Done")


if __name__ == "__main__":
    main()
