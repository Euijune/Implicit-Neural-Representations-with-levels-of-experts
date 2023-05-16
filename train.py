import argparse
import torch
from torch.utils.data import DataLoader

import os
import imageio
import numpy as np

from datetime import datetime
import random

from data.dataset import ImageFitting
from model.model import MLP
from utils.metric import calculate_PSNR, calculate_SSIM

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='cameraman', help='save path of image for INR traning')
    parser.add_argument('--sidelen', type=int, default=256, help='image width or height')
    parser.add_argument('--mgrid_min', type=int, default=0, help='2d coord -> 1d coord min value')
    parser.add_argument('--mgrid_max', type=int, default=1, help='2d coord -> 1d coord max value')
    parser.add_argument('--tile_size', type=int, default=4, help='weight tile size, tile size N^2 = N X N tile size')
    parser.add_argument('--hidden_features', type=int, default=64, help='hidden layer output channel')
    parser.add_argument('--hidden_layers', type=int, default=9, help='number of hidden layer')
    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--coord_batch_size', type=int, default=256*256, help='coordinate batch size')

    parser.add_argument('--epoch', type=int, default=1001, help='number of training steps')
    parser.add_argument('--steps_til_summary', type=int, default=50, help='show progress image per steps_til_summary epoch')

    parser.add_argument('--save_dir', type=str, help='The path where the experiment results are stored')
    parser.add_argument('--model_name', type=str, help='model name')
    parser.add_argument('--weight_path', type=str, help='if weight_path is none, training model from scratch')

    return parser.parse_known_args()[0] if known else parser.parse_args()


def train(opt):
    if opt.model_name is None:
        print("You must set the model name")
        raise ValueError

    dir = opt.save_dir
    if not os.path.exists(dir):
        os.makedirs(dir)

    try:
        os.makedirs(f'{dir}/{opt.model_name}_ckpt')
        os.makedirs(f'{dir}/{opt.model_name}_output')
    except FileExistsError:
        pass

    print("\n######################## Training ########################")
    img = ImageFitting(img_path=opt.img_path, sidelength=opt.sidelen, mgrid_min=opt.mgrid_min, mgrid_max=opt.mgrid_max, normalize=True)
    dataloader = DataLoader(img, batch_size=1, pin_memory=True, num_workers=0)

    img_LoE = MLP(tile_size=opt.tile_size, input_dim=2, hidden_features=opt.hidden_features, 
                    hidden_layers=opt.hidden_layers, output_dim=1, grid_arrangement_str='fine_to_coarse', 
                    coord_batch_size=opt.coord_batch_size)
    img_LoE.cuda()
    if opt.weight_path is not None:
        img_LoE.load_state_dict(torch.load(f'{opt.weight_path}'))

    optim = torch.optim.Adam(lr=opt.lr, params=img_LoE.parameters(), betas=(0.9, 0.995))
    scheduler = None
    if opt.epoch > 5000:
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=5000, gamma=0.1)

    model_input, ground_truth = next(iter(dataloader))  # input = coords, gt = img
    model_input, ground_truth = model_input.squeeze(0).cuda(), ground_truth.squeeze(0).cuda()

    start_time = datetime.now()

    best_loss = 99999.9
    for step in range(opt.epoch):
        model_output = img_LoE(model_input)    
        loss = ((model_output - ground_truth)**2).mean()

        optim.zero_grad()
        loss.backward()
        optim.step()

        if scheduler is not None:
            scheduler.step()

        # Verbose
        if not step % opt.steps_til_summary:

            s = model_output.cpu().view(256, 256).detach().numpy()
            img_source = ((s - s.min()) * (1/(s.max() - s.min()) * 255)).astype('uint8')    # [-1, 1] -> [0, 255]

            t = ground_truth.cpu().view(256, 256).detach().numpy()
            img_target = ((t - t.min()) * (1/(t.max() - t.min()) * 255)).astype('uint8')    # [-1, 1] -> [0, 255]

            psnr = calculate_PSNR(img_source, img_target)
            score, _ = calculate_SSIM(img_source, img_target)

            print(f"Step {step}, Total loss {loss:.6f}, PSNR {psnr:.6f}, SSIM {score:.6f}")
            print(f'time elapsed: {datetime.now() - start_time}\n')

            imageio.imsave(f'{dir}/{opt.model_name}_output/{step}.png', img_source)

        # Save model weight
        if loss < best_loss:
            best_loss = loss
            torch.save(img_LoE.state_dict(), f"{dir}/{opt.model_name}_ckpt/{opt.model_name}_best.pth")
            s = model_output.cpu().view(256, 256).detach().numpy()
            img_source = ((s - s.min()) * (1/(s.max() - s.min()) * 255)).astype('uint8')
            imageio.imsave(f'{dir}/{opt.model_name}_output/best.png', img_source)

        if step == (opt.epoch - 1):
            torch.save(img_LoE.state_dict(), f"{dir}/{opt.model_name}_ckpt/{opt.model_name}_last.pth")                     
    
    print("######################## Done ########################\n")


def main(opt):
    seed_everything(37) # Seed 고정
    train(opt)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)