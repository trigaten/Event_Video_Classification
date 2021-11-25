from EADataset import EADataset
from torch.utils.data import DataLoader
from Net import Net
from train import train
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from statistics import mean 
import torch
import gc

# log values to tensorboard during training
writer = SummaryWriter("LOOcross_val_new")
root_dir = "train"
# models generally converge at 60 epochs
EPOCHS = 20
loss_func = CrossEntropyLoss()
global_dataset = EADataset(root_dir, 'cuda')
device = 'cuda'

cs_losses = []
cs_preds = []
cs_acc = []


for index in range(len(global_dataset)):
    print(index)
    paths = global_dataset.video_paths.copy()
    paths.pop(index)
    dataset = EADataset(root_dir, device, video_paths=paths, do_slice=True, do_common=True)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    net = Net(len(dataset.actions), device=device).to(device)
    # print(torch.cuda.memory_summary(abbreviated=True))

    train(dataloader, net, EPOCHS, writer=writer, loss_log_path="Loss: " + str(index), log_mem_path=str(index))

    with torch.no_grad():
        one_left_out, label = global_dataset[index]

        logits = net(one_left_out)
        
        last = logits[len(logits)-1]
        last = last.unsqueeze(0)
        
        loss = loss_func(last, label.unsqueeze(0))
        cs_losses.append(loss.detach().item())
        cs_preds.append(torch.argmax(last.detach()).item())

        correct = torch.argmax(last).item() == label.item()

        cs_acc.append(correct)

        writer.add_scalar("CV loss", loss.detach().item(), index)
        writer.add_scalar("CV pred", torch.argmax(last).detach().item(), index)
        writer.add_scalar("CV acc", correct, index)

        # del dataloader
        # del net
        # del logits
        # gc.collect()
        # torch.cuda.empty_cache()


    
    # print("HERE")

print(cs_preds)
print(mean(cs_losses))
print(mean(cs_acc))


# [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 0, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1]
# 0.937792854813429
# 0.9807692307692307