import datetime
import logging
import os
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import cv2
from PIL import Image
from ssd.engine.inference import do_evaluation
from ssd.utils import distributed_util


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = distributed_util.get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k, v in loss_dict.items():
            loss_names.append(k)
            all_losses.append(v)
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def _save_model(logger, model, model_path,iteration=None):
    if isinstance(model, DistributedDataParallel):
        vgg_model = model.module
    torch.save({
        'iteration': iteration+1,
        'state_dict': model.state_dict(),
    }, model_path)
    logger.info("Saved checkpoint to {}".format(model_path))


def do_train(cfg, model,
             data_loader,
             optimizer,
             scheduler,
             device,
             args,
             resume_iteration=0):
    logger = logging.getLogger("SSD.trainer")
    logger.info("Start training")
    model.train()
    save_to_disk = distributed_util.get_rank() == 0
    if args.use_tensorboard and save_to_disk:
        import tensorboardX

        summary_writer = tensorboardX.SummaryWriter(log_dir=cfg.OUTPUT_DIR)
    else:
        summary_writer = None

    max_iter = len(data_loader)
    start_training_time = time.time()
    trained_time = 0
    tic = time.time()
    end = time.time()
    import numpy as np
    count_useful_iteration = 0
    count_not_useful_iteration = 0
    for iteration, (images, quads, labels, score_map) in enumerate(data_loader):
        #size infoes
        #shape(boxes):[2, 24564, 8]
        #shape(labels):[2, 24564]
        #shape(images):[2, 3, 512, 512]
        #print('iteration:',iteration)
        if args.resume:
            iteration = resume_iteration+iteration
        else:
            iteration = iteration + 1

        if save_to_disk and iteration % args.save_step == 0:
            model_path = os.path.join(cfg.OUTPUT_DIR,"ssd{}_vgg_iteration_{:06d}.pth".format(cfg.INPUT.IMAGE_SIZE, iteration))
            _save_model(logger, model, model_path, iteration)
        scheduler.step()
        # labels_temp = labels.numpy()
        # labels_temp = np.squeeze(labels_temp,0)
        # index = np.squeeze(np.argwhere(labels_temp == 1),1)
        # print('index:',index)
        # current_quad = quads[0,index,:]
        # print(current_quad)
        # temp_img = images.numpy()[0]
        # temp_img = np.swapaxes(temp_img, 0, 1)
        # temp_img = np.swapaxes(temp_img, 1, 2)
        # for i in range(np.shape(current_quad)[0]):
        #     cv2.circle(temp_img, (int(current_quad[i][0]), int(current_quad[i][1])), 5, (0, 255, 0), 5)
        #     cv2.circle(temp_img, (int(current_quad[i][2]), int(current_quad[i][3])), 5, (255, 255, 255), 5)
        #     cv2.circle(temp_img, (int(current_quad[i][4]), int(current_quad[i][5])), 5, (255, 0, 0), 5)
        #     cv2.circle(temp_img, (int(current_quad[i][6]), int(current_quad[i][7])), 5, (0, 0, 255), 5)
        # cv2.imshow('img', temp_img.astype(np.uint8))
        # cv2.waitKey()
        if len(quads) == 0:
            print('quads is None')
            continue
        images = images.to(device)
        quads = quads.to(device)

        labels = labels.to(device)

        num_pos = torch.sum(labels)
        if num_pos == 0:
            count_not_useful_iteration += 1
            print('num_pos==0 and no pos sample found')
            continue
        else:
            # print(num_pos)
            count_useful_iteration += 1
        optimizer.zero_grad()
        if score_map is None:
            loss_dict = model(images, targets=(quads, labels))
        else:
            score_map = score_map.to(device)
            loss_dict = model(images,(quads, labels), score_map)



        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss = sum(loss for loss in loss_dict.values())
        loss.backward()
        optimizer.step()
        trained_time += time.time() - end
        end = time.time()
        if iteration % args.log_step == 0:
            eta_seconds = int((trained_time / iteration) * (max_iter - iteration))
            log_str = [
                "Iter: {:06d}, Lr: {:.7f}, Cost: {:.2f}s, Eta: {}".format(iteration,
                                                                          optimizer.param_groups[0]['lr'],
                                                                          time.time() - tic, str(datetime.timedelta(seconds=eta_seconds))),
                "total_loss: {:.3f}".format(losses_reduced.item())
            ]
            log_str.append("{}: {:.3f}".format('regression_loss', loss_dict_reduced['regression_loss']))
            log_str.append("{}: {:.6f}".format('classification_loss', loss_dict_reduced['classification_loss']))
            log_str.append("{}: {:.5f}".format('fcn_loss', loss_dict_reduced['fcn_loss']))

            log_str = ', '.join(log_str)
            logger.info(log_str)
            if summary_writer:
                global_step = iteration
                summary_writer.add_scalar('losses/total_loss', losses_reduced, global_step=global_step)
                for loss_name, loss_item in loss_dict_reduced.items():
                    summary_writer.add_scalar('losses/{}'.format(loss_name), loss_item, global_step=global_step)
                summary_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)

            tic = time.time()
        # Do eval when training, to trace the mAP changes and see performance improved whether or nor
        # if args.eval_step > 0 and iteration % args.eval_step == 0 and not iteration == max_iter:
        #     do_evaluation(cfg, model, cfg.OUTPUT_DIR, distributed=args.distributed)
        #     model.train()

    # if save_to_disk:
    #     model_path = os.path.join(cfg.OUTPUT_DIR, "ssd{}_vgg_final.pth".format(cfg.INPUT.IMAGE_SIZE))
    #     _save_model(logger, model, model_path)
    # compute training time
    total_training_time = int(time.time() - start_training_time)
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info("Total training time: {} ({:.4f} s / it)".format(total_time_str, total_training_time / max_iter))
    with open('useful_iteration.txt','w') as f:
        f.write('count_useful_iteration:'+str(count_useful_iteration)+'\n')
        f.write('count_not_useful_iteration:'+str(count_not_useful_iteration))




    return model
