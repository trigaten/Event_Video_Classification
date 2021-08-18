from EADataset import EADataset
from torch.utils.data import DataLoader
from Net import Net
from train import train
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from statistics import mean 
import torch

# log values to tensorboard during training
writer = SummaryWriter("LOOcross_val")
root_dir = "train"
# models generally converge at 60 epochs
EPOCHS = 60
loss_func = CrossEntropyLoss()
global_dataset = EADataset(root_dir, 'cuda')
device = 'cuda'

cs_losses = []
cs_preds = []
cs_acc = []


for index in range(len(global_dataset)):
    paths = global_dataset.video_paths.copy()
    paths.pop(index)
    dataset = EADataset(root_dir, device, video_paths=paths, do_slice=True)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    net = Net(len(dataset.actions), device=device).to(device)

    train(dataloader, net, EPOCHS, writer=writer, loss_log_path="Loss: " + str(index))

    one_left_out, label = global_dataset[index]

    logits = net(one_left_out)
    
    last = logits[len(logits)-1]
    last = last.unsqueeze(0)
    label = label.unsqueeze(0)
    
    loss = loss_func(last, label)
    cs_losses.append(loss.item())
    cs_preds.append(torch.argmax(last).item())

    _, exp = global_dataset[index]
    correct = torch.argmax(last).item() == exp.item()
    cs_acc.append(correct)
    
    writer.add_scalar("CV loss", loss, index)
    writer.add_scalar("CV pred", torch.argmax(last).item(), index)
    writer.add_scalar("CV acc", correct, index)

    
    # print("HERE")

print(cs_preds)
print(mean(cs_losses))
print(mean(cs_acc))


