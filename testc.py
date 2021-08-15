import torch
import torch
# import torch.nn.functional as F
# t4d = torch.Tensor([[1, 0, 0]])
# p1d = (2, 2) # pad last dim by 1 on each side
# out = F.pad(t4d, p1d, "constant", 1) 

# print(out)

import torch.nn.functional as F
# source = torch.zeros((5,10))
# now we expand to size (7, 11) by appending a row of 0s at pos 0 and pos 6, 
# and a column of 0s at pos 10
# result = F.pad(input=source, pad=(0, 1, 1, 1), mode='replicate', value=0)

# print(source.shape)

exp_out_tensor = torch.zeros(5)
exp_out_tensor[3] = 1

exp_out_tensor = torch.unsqueeze(exp_out_tensor, 0)

print(exp_out_tensor.repeat(2, 1))

print(torch.empty(3, dtype=torch.long).random_(5))