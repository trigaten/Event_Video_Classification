"""
This code trains the model

Training procedure:
for each epoch:
    for each sample video in the train dataset (sampled randomly)
        push the entire video through the neural network and get a class prediction tensor for each frame
        apply loss function using the label of the video as the target class for (almost) every frame 
        get average loss from loss function and perform SGD. The first 20 frames are left out of the
        loss calculation since the network does not neccesarily have enough data to make a decision
        before this point.

This method basically treats each video as a batch of images since the loss is their average loss
Note, however, that the prediction of each frame/image is affected by the previous images due to
the LSTM in the model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.io as IO
import sys

NO_LOSS_FRAMES = 0#20

def train(dataloader, net, epochs, writer=None, model_save_path=None, loss_log_path="Loss", log_mem_path=None):
    # initialize loss function and optimizer
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    
    # keep track of number of videos that have been trained on
    video_num = 0
    mean_acc = 0
    acc = 0

    for epoch in range(epochs):
        for index, sample in enumerate(dataloader):
            # exp is expected class
            video, exp = sample
            # remove first dim from video
            video = video.squeeze(0)
            
            # remove first 2 dims from exp
            exp = exp.squeeze(0)
            exp = exp.squeeze(0)

            # make exp into a 1d tensor which contains len(video) copies of exp
            # this is necessary for the loss function which accepts all the video frame
            # output tensors and expects a ground truth value for each frame
            rep_exp = exp.repeat(len(video)-NO_LOSS_FRAMES)

            # push video through neural net
            out = net(video)
            # if log_mem_path:
            #     writer.add_scalar("ten storage" + log_mem_path, sys.getsizeof(out.storage()), video_num)
            #     writer.add_scalar("mem available" + log_mem_path, (torch.cuda.memory_reserved(0)-torch.cuda.memory_allocated(0)), video_num)

            out = out[NO_LOSS_FRAMES:]

            
            # compute loss
            loss_out = loss(out, rep_exp)

            # print(loss_out)

            # accumulate gradients
            loss_out.backward()

            # perform SGD
            optimizer.step()

            # clear grads
            optimizer.zero_grad()
           
            # log loss
            writer.add_scalar(loss_log_path, loss_out.detach().item(), video_num)
            # writer.add_scalar("epoch" + str(epoch), epoch, epoch)
            # writer.add_scalar("label" + str(epoch), exp[0].detach().item(), video_num)

            if torch.argmax(torch.mean(out, dim=0)).item() == exp.item():
                mean_acc+=1
            if torch.argmax(out[len(out)-1]).item() == exp.item():
                acc+=1
            # print(torch.argmax(out[len(out)-1]))
            # print(exp)
            # print(torch.mean(out, dim=0))
            # print(out[len(out)-1])
            writer.add_scalar("mean_acc", mean_acc, video_num)
            writer.add_scalar("acc", acc, video_num)
            
            # del loss_out
            # del out 
            # del video
            # del exp
            # gc.collect()
            # torch.cuda.empty_cache()
            # if loss_out > 1.7 and epoch > 1:
            #     v = video.permute(0, 2, 3, 1).detach().cpu()
            #     r = torch.zeros([v.shape[0], v.shape[1], v.shape[2], 1])
            #     v = torch.cat((r, v), 3)       
            #     print(video.shape)
            #     print(exp[0])

            #     IO.write_video("savids/video" + str(epoch) + "_" + str(index) + ".avi", v.numpy(), 30.0)
        # return False
            video_num+=1

        # save model every 10 epochs
        # if model_save_path and epoch % 10 == 0:
        #     print("Saving")
        #     torch.save(net.state_dict(), model_save_path)
        # gc.collect()

    writer.flush()

    # save after all training done
    if model_save_path:
            torch.save(net.state_dict(), model_save_path)