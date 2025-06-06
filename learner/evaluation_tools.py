import yaml
import import_ipynb
from model import Model
import torch
from os.path import join as opj
import sys
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from base_model import *
import matplotlib.pyplot as plt
import time
import os
import imageio

base_path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(opj(base_path,'utils'))
from ev_utils import *
import matplotlib
matplotlib.use('agg')

def eval_plotter(learner, checkpoint_path=None, load_ckpt=False, dataSetstoTest=5):

    # load particular checkpoint if specified, otherwise use current weights
    if load_ckpt:
        model_ckpt = os.path.splitext(os.path.basename(checkpoint_path[0]))[0]
        title = model_ckpt
        learner.Model.load_from_checkpoint(checkpoint_path)
        learner.mylogger(f'[EVAL_TOOLS] Evaluating {title}...')
    else:
        title="Evluation"

    numtrajs_plotting = min(dataSetstoTest, learner.num_val_steps)
    # Logger Arrays
    ims = []
    evframes = []
    preds = []
    cmds = []
    # Eval Loop
    with torch.no_grad():

        # starting index of trajectories in dataset
        # train
        train_traj_starts = np.cumsum(learner.train_data["traj_lengths"]) - learner.train_data["traj_lengths"]
        train_traj_starts = np.hstack((train_traj_starts, -1))
        # val
        val_traj_starts = np.cumsum(learner.val_data["traj_lengths"]) - learner.val_data["traj_lengths"]
        val_traj_starts = np.hstack((val_traj_starts, -1))

        ### Validation loop ###
        learner.Model.model.eval()

        st_run = time.time()

        num_trains = 1

        traj_names = []

        for it in range(numtrajs_plotting):

            # return (loss, loss_terms), (return_pred(preds), extras), (traj_input_ims, traj_input_evs, traj_input_desvels, gt)

            if it < num_trains:
                loss, (pred, extras), (traj_input_ims, traj_input_evs, desvel, gt) = learner.run_model(it, train_traj_starts, learner.train_data["traj_lengths"], np.arange(len(train_traj_starts)), 'train', return_inputs=True, do_step=False)
                # save last part of path to list
                traj_names.append('train_'+os.path.basename(learner.train_data["traj_folders"][it]))
            else:
                loss, (pred, extras), (traj_input_ims, traj_input_evs, desvel, gt) = learner.run_model(it, val_traj_starts,learner.val_data["traj_lengths"] , np.arange(len(val_traj_starts)), 'val', return_inputs=True, do_step=False)
                traj_names.append('val_'+os.path.basename(learner.val_data["traj_folders"][it]))

            if  learner.model_type == 'OrigUnet' or learner.model_type=='OrigUnet_lstm':

                (pred_vel, pred), pred_upconv = pred, extras[1]
                pred_vel = pred_vel * desvel.cpu()
                cmd = gt[0]

            else: # not recognized
                learner.logger.info(f'[EVAL_TOOLS] model_type {learner.model_type} not recognized for visualization')

            # save first non-free space image of traj for viewability
            if traj_input_ims.mean() < 0.95:
                first_non_blank_index = np.where(np.mean(traj_input_ims.cpu().detach().numpy(), axis=(2, 3)) < 0.90)[0][0]
            else:
                first_non_blank_index = 0
            first_non_blank_image = traj_input_ims[first_non_blank_index, 0]
            ims.append(first_non_blank_image.cpu().detach().numpy())
            preds.append(pred_vel.cpu().detach().numpy())
            cmds.append(cmd.cpu().detach().numpy())
            if traj_input_evs is not None:
                first_non_blank_image_evframe = traj_input_evs[first_non_blank_index, 0]
                evframes.append(first_non_blank_image_evframe.cpu().detach().numpy())
            else:
                evframes.append(np.zeros_like(ims[-1]))

        learner.logger.info(f'[EVAL_TOOLS] Evaluated {numtrajs_plotting} trajectories in {time.time() - st_run:.2f} s')

    st_fig = time.time()

    # scale up evs if needed
    if learner.rescale_evs > 0.0:
        evframes = [evframe*learner.rescale_evs for evframe in evframes]

    ##Plotter##
    # TODO with long trajectories (200 samples) plotting the preds/cmds is very slow for the first plot (40s) then short for subsequent (4s)
    fig, axs = plt.subplots(5, numtrajs_plotting, figsize=(16, 8))
    for i in range(numtrajs_plotting):
        axs[0, i].imshow(ims[i])
        axs[0, i].set_title(traj_names[i])
        axs[1, i].imshow(evframes[i])
        axs[2, i].plot(cmds[i][:, 0], label='gt')

        axs[3, i].plot(preds[i][:, 1], label='pred', marker='.')
        axs[3, i].plot(cmds[i][:, 1], label='gt')
        axs[3, i].set_ylim([-np.max(np.abs(cmds[i][:, 1]))-.5, np.max(np.abs(cmds[i][:, 1]))+.5])

        axs[4, i].plot(preds[i][:, 2], label='pred', marker='.')
        axs[4, i].plot(cmds[i][:, 2], label='gt')
        axs[4, i].set_ylim([-np.max(np.abs(cmds[i][:, 2]))-.5, np.max(np.abs(cmds[i][:, 2]))+.5])

        if i == 0:
            axs[2, i].legend()
            axs[3, i].legend()
            axs[4, i].legend()
            axs[0, i].set_ylabel(f"sample image")
            axs[1, i].set_ylabel(f"sample evframe")
            axs[2, i].set_ylabel(f"x vel")
            axs[3, i].set_ylabel(f"y vel")
            axs[4, i].set_ylabel(f"z vel")

    fig.suptitle(title)

    learner.logger.info(f'[EVAL_TOOLS] Plotted {numtrajs_plotting} trajectories in {time.time() - st_fig:.2f} s')

    return fig, title

def visualize_images(learner, checkpoint_path=None, load_ckpt=False, dataSetstoTest=5):
    
    if load_ckpt:
    # load particular checkpoint if specified, otherwise use current weights
        model_ckpt = os.path.splitext(os.path.basename(checkpoint_path[0]))[0]
        title = model_ckpt
        learner.Model.load_from_checkpoint(checkpoint_path)
        learner.logger.info(f'[EVAL_TOOLS] Evaluating {title}...')
    else:
        title="Evaluation"
    # Eval Loop
    with torch.no_grad():

        # starting index of trajectories in dataset
        # train
        train_traj_starts = np.cumsum(learner.train_data["traj_lengths"]) - learner.train_data["traj_lengths"]
        train_traj_starts = np.hstack((train_traj_starts, -1))
        # val
        val_traj_starts = np.cumsum(learner.val_data["traj_lengths"]) - learner.val_data["traj_lengths"]
        val_traj_starts = np.hstack((val_traj_starts, -1))

        ### Validation loop ###
        learner.Model.model.eval()

        st_run = time.time()

        num_evals = 3
        num_trains = 1

        traj_names = []
        traj_output = []

        for it in range(num_evals):
            
            if it < num_trains:

                traj_starts = train_traj_starts
                trajlength = learner.train_data["traj_lengths"]
                dirs = learner.train_data["traj_folders"]
                mode = 'train'

            else:

                traj_starts = val_traj_starts
                trajlength = learner.val_data["traj_lengths"]
                dirs = learner.val_data["traj_folders"]
                mode = 'val'

            # run model
            loss, (pred, extras), (traj_input_ims, traj_input_evs, desvel, gt) = learner.run_model(it if mode=='train' else it-num_trains, traj_starts, trajlength, np.arange(len(traj_starts)), mode, return_inputs=True, do_step=False)
            
            # scale up evs if needed
            if learner.rescale_evs > 0.0:
                traj_input_evs *= learner.rescale_evs

            # save desired outputs
            traj_names.append('train_'+os.path.basename(dirs[it if mode=='train' else it-num_trains]))
            
            if  learner.model_type == 'OrigUNet' or learner.model_type=='OrigUNet_lstm':

                (pred_vel, pred), pred_upconv = pred, extras[1]

            else: # not recognized

                learner.logger.info(f'[EVAL_TOOLS] model_type {learner.model_type} not recognized for visualization')

            traj_output.append((traj_input_evs, pred_upconv, pred, pred_vel, gt, desvel))

        # now save two gifs, one for train and one for val
        # where the four parts of train/val output are tiled into single images then gif'd consecutively
        for traj_i in range(len(traj_output)):
            h, w = traj_output[0][0].shape[2], traj_output[0][0].shape[3]
            gif = []
            for i in range(traj_output[traj_i][0].shape[0]):
                
                frame = np.zeros((2*h, 2*w, 3), dtype=np.uint8)
                
                evfr, _ = simple_evim(traj_output[traj_i][0][i].cpu().detach().numpy().squeeze(), style='redblue-on-white')
                
                # expand upconv'd images up to almost hxw
                pred_upconv = traj_output[traj_i][1][i].cpu().detach().numpy().squeeze()
                pred_upconv = np.kron(pred_upconv, np.ones((h//pred_upconv.shape[0], w//pred_upconv.shape[1])))

                frame[:h, :w] = evfr

                # for depth prediction, make values >1.0 clipped to 1.0
                pred_upconv = np.clip(pred_upconv, 0.0, 1.0)

                frame[:pred_upconv.shape[0], w:w+pred_upconv.shape[1]] = (np.stack([pred_upconv] * 3, axis=-1) * 255).astype(np.uint8)
                pred_im = traj_output[traj_i][2][i].cpu().detach().numpy().squeeze()

                # again, clip to 1.0 for depth
                pred_im = np.clip(pred_im, 0.0, 1.0)

                pred_im = (np.stack([pred_im] * 3, axis=-1) * 255).astype(np.uint8)
                gt_im = (np.stack([traj_output[traj_i][4][1][i].cpu().detach().numpy().squeeze()] * 3, axis=-1) * 255).astype(np.uint8)

                # draw arrow from center of image according to velocity pred
                yvel = traj_output[traj_i][3][i].cpu().detach().numpy().squeeze()[1]
                zvel = traj_output[traj_i][3][i].cpu().detach().numpy().squeeze()[2]
                # NOTE comment out next line if don't want arrow for velocity prediction
                pred_im = cv2.arrowedLine(pred_im, 
                                               (pred_im.shape[1]//2, pred_im.shape[0]//2), 
                                               (int(pred_im.shape[1]//2 - yvel*min(pred_im.shape[0:2])), 
                                                int(pred_im.shape[0]//2 - zvel*min(pred_im.shape[0:2]))), 
                                               (0, 0, 255), 2)
                frame[h:, :w] = pred_im
                
                # similarly, draw error with gt velocity command for comparison
                yvel = traj_output[traj_i][4][0][i].cpu().detach().numpy().squeeze()[1] / traj_output[traj_i][5][i].cpu().detach().numpy().squeeze()
                zvel = traj_output[traj_i][4][0][i].cpu().detach().numpy().squeeze()[2] / traj_output[traj_i][5][i].cpu().detach().numpy().squeeze()
                # NOTE comment out next line if don't want arrow for velocity gt
                gt_im = cv2.arrowedLine(gt_im, 
                                              (gt_im.shape[1]//2, gt_im.shape[0]//2), 
                                              (int(gt_im.shape[1]//2 - yvel*min(gt_im.shape[0:2])), 
                                               int(gt_im.shape[0]//2 - zvel*min(gt_im.shape[0:2]))), 
                                              (0, 0, 255), 2)
                frame[h:, w:] = gt_im

                gif.append(frame)

            if len(traj_output) > 0:
                mode = 'train' if traj_i < num_trains else 'val'
                learner.logger.info(f'[EVAL_TOOLS] {mode} traj output idx {traj_i} has {len(gif)} frames')
                # convert to gif
                gif = np.stack(gif)
                imageio.mimsave(opj(learner.workspace, f"{os.path.basename(learner.workspace)}__{title}_{mode}{traj_i if mode=='train' else traj_i-num_trains}.gif"), gif)
            else:
                learner.logger.info(f'[EVAL_TOOLS] traj output idx {traj_i} empty')

if __name__ == '__main__':

    from learner import Learner

    dataSetstoTest = 5 # Must be less than or equal to the number of validation trajectories

    # Setup
    
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # load model with checkpoint
    with open(opj(base_path,"configs/test.yaml")) as f:
        test_config = yaml.safe_load(f)
    learner=Learner(test_config)
    st_time = time.time()

    fig, title = eval_plotter(learner, dataSetstoTest=dataSetstoTest, load_ckpt=False)
    fig.savefig(opj(learner.workspace, f'eval_{os.path.basename(learner.workspace)}__{title}.png'))
    learner.logger.info(f'[EVAL_TOOLS] eval_plotter finished {title} in {time.time() - st_time:.2f} s')

    st_time = time.time()
    visualize_images(learner, dataSetstoTest=dataSetstoTest, load_ckpt=False,device='cpu')
    learner.logger.info(f'[EVAL_TOOLS] visualize_images finished in {time.time() - st_time:.2f} s')

