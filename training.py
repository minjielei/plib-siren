'''Implements a generic training loop.
'''

import torch
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
import os
import shutil
import utils

def train(model, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn, loss_schedules=None):

    optim = torch.optim.AdamW(lr=lr, params=model.parameters(), amsgrad=True)

    epoch_start = 0
    total_steps = 0
    train_losses = []
    
    if os.path.exists(model_dir+'/checkpoints'):
        val = input("The model directory %s exists. Load latest run? (y/n)" % model_dir)
        if val == 'y':
            filename = utils.find_latest_checkpoint(model_dir)
            model, optim, total_steps, epoch_start, train_losses =  utils.load_checkpoint(model, optim, filename)
            print(optim.param_groups[0]['lr'])
            if val == 'n':
                shutil.rmtree(model_dir)

    summaries_dir = os.path.join(model_dir, 'summaries')
    if not os.path.exists(summaries_dir):
        os.makedirs(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    writer = SummaryWriter(summaries_dir)
    
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor = 0.5, patience=100, min_lr=5e-6)

    for epoch in range(epoch_start, epochs):

        for step, (model_input, gt) in enumerate(train_dataloader):
            
            model_input = {key: value.cuda() for key, value in model_input.items()}
            gt = {key: value.cuda() for key, value in gt.items()}
            model_output = model(model_input['coords'])
            
            losses = loss_fn(model_output, gt, model_input['weights'])
    
            train_loss = 0.
            for loss_name, loss in losses.items():
                single_loss = loss.mean()

                if loss_schedules is not None and loss_name in loss_schedules:
                    writer.add_scalar(loss_name + "_weight", loss_schedules[loss_name](total_steps), total_steps)
                    single_loss *= loss_schedules[loss_name](total_steps)

                writer.add_scalar(loss_name, single_loss, total_steps)
                train_loss += single_loss

            train_losses.append(train_loss.item())
            writer.add_scalar("total_train_loss", train_loss, total_steps)

            if not total_steps % steps_til_summary:
                torch.save({
                        'step': total_steps,
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optim.state_dict(),
                        'loss': train_losses,
                        },  os.path.join(checkpoints_dir, 'model_current.pth'))

            optim.zero_grad()
            train_loss.retain_grad()
            train_loss.backward()
            optim.step()
#             scheduler.step(train_loss)

            total_steps += 1

        if not epoch % epochs_til_checkpoint and epoch:
            print('epoch:', epoch )

            torch.save({
                        'step': total_steps,
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optim.state_dict(),
                        'loss': train_losses,
                        },  os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
            
            np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch),
                       np.array(train_losses))

            plt_name = os.path.join(checkpoints_dir, 'total_loss_epoch_%04d.png' % epoch)
            utils.plot_losses(total_steps, train_losses, plt_name)

#             ground_truth = gt['coords'].cpu().detach().numpy()[:, 45]
#             pred = model_output['model_out'].cpu().detach().numpy()[:, 45]
#             plt_name = os.path.join(checkpoints_dir, 'pred_vs_truth_epoch_%04d.png' % epoch)
#             utils.plot_pred_vs_gt(pred, ground_truth, plt_name)

    torch.save(model.state_dict(),
               os.path.join(checkpoints_dir, 'model_final.pth'))
    np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
               np.array(train_losses))
    
    #Plot and save loss
    plt_name = os.path.join(model_dir, 'total_loss.png')
    utils.plot_losses(total_steps, train_losses, plt_name)

    # Make images and plots
#     ground_truth = gt['coords'].cpu().detach().numpy()[:, 45]
#     pred = model_output['model_out'].cpu().detach().numpy()[:, 45]
    # ground_truth_video = np.reshape(ground_truth, (data_shape[0], data_shape[1], data_shape[2], -1))
    # predict_video = np.reshape(pred, (data_shape[0], data_shape[1], data_shape[2], -1))
    
#     utils.plot_pred_vs_gt(pred, ground_truth, os.path.join(model_dir, 'pred_vs_truth.png'))
#     utils.plot_hist_overlap(pred, ground_truth, os.path.join(model_dir, 'asym.png'))  
    # utils.draw_img(predict_video, ground_truth_video, model_dir)