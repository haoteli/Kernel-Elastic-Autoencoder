from torch.autograd import Variable

import torch



def K(x, y): # This computes the kernel between x and y
    x_size = x.shape[0]
    y_size = y.shape[0]
    dim = x.shape[1]
    tiled_x = x.view(x_size,1,dim).repeat(1, y_size,1)
    tiled_y = y.view(1,y_size,dim).repeat(x_size, 1,1)

    return torch.exp(-torch.mean((tiled_x - tiled_y)**2,dim=2)/(dim*0.0005)) # [batch_size, 1]
    

def mmd(x, y): # The MMD loss between x and y

    #return torch.mean(K(x, x)) + torch.mean(K(y, y)) - 2*torch.mean(K(x, y))
    #return torch.mean(K(x, x)) - 2*torch.mean(K(x, y))
    return 1*(1 - torch.mean(K(x, y)))



def train(model, device, train_batch, optimizer, criterion, clip, n_cat, bottle_len,HID_DIM):
    
    model.train()
    
    
    for i, (inputs, _) in enumerate(train_batch):
        data = inputs
        #data = train_batch[i]
        src = data.to(device)
        trg = data[:,:-n_cat].long().to(device)
        
        
        optimizer.zero_grad()
        
        outputs = model(src, trg[:,:-1]) # The second output is the attention
        output = outputs[0]
        
        outputs_no_noise = model(src, trg[:,:-1], use_noise = False) # The second output is the attention
        output_no_noise = outputs_no_noise[0]
        
        latent = outputs[-1]
   
        
        output_dim = output.shape[-1]
        
        output = output.view(-1, output_dim)
        output_no_noise = output_no_noise.view(-1, output_dim)
        
        trg = trg[:,1:].reshape(-1)
        
        mmd_loss =  mmd(latent,Variable(torch.randn(1000,bottle_len*HID_DIM).to(device)))
      
        rec_loss = torch.mean(criterion(output, trg))
        rec_loss_no_noise =  torch.mean(criterion(output_no_noise, trg))
       
        loss =  mmd_loss + (rec_loss + 2*rec_loss_no_noise)/3 #+ (1)*prop_pred_loss

        if(i%25 == 0):
          print('Loss:',loss.item())
          print('Rec Loss ',rec_loss.item())
        
          print('MMD LOSS:', mmd_loss.item())
          print('Rec Loss No Noise',rec_loss_no_noise.item())
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
