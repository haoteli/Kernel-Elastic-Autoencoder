import torch
import torch.nn as nn
import torch.nn.functional as F



class Seq2Seq(nn.Module):
    def __init__(self, 
                 encoder, 
                 decoder, 
                 src_pad_idx, 
                 trg_pad_idx, 
                 device,
                 bottle_len,
                 data_maxlength
                 ):
        super().__init__()
        self.bottle_len = bottle_len
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
        self.tanh = nn.Tanh()
        maxlength = data_maxlength
        self.maxlength = data_maxlength
        self.latent = nn.Linear(in_features = (maxlength), out_features = bottle_len, bias = True)

        self.inverse = nn.Linear(in_features = (bottle_len+self.encoder.n_cat)*self.decoder.hid_dim, out_features = bottle_len*self.decoder.hid_dim)
        self.l1 = nn.Linear(self.bottle_len*self.decoder.hid_dim,self.bottle_len*self.decoder.hid_dim)
        
        self.l2 = nn.Linear(self.bottle_len*self.decoder.hid_dim, self.encoder.n_cat)
        
        self.relu = nn.ReLU()
    

    def make_src_mask(self, src):
        
        #src = [batch size, src len]
        
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        #src_mask = [batch size, 1, 1, src len]

        return src_mask
    
    def make_trg_mask(self, trg):
        
        #trg = [batch size, trg len]
        
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        
        #trg_pad_mask = [batch size, 1, 1, trg len]
        
        trg_len = trg.shape[1]
        
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = trg.device)).bool()
        
        #trg_sub_mask = [trg len, trg len]
            
        trg_mask = trg_pad_mask & trg_sub_mask
        
        #trg_mask = [batch size, 1, trg len, trg len]
        
        return trg_mask

    
    def forward(self, src, trg, use_noise = True):
        
        #src = [batch size, src len]
        #trg = [batch size, trg len]
                
        src_mask = self.make_src_mask(src.long())
        trg_mask = self.make_trg_mask(trg)
        
        #src_mask = [batch size, 1, 1, src len]
        #trg_mask = [batch size, 1, trg len, trg len]
        
        enc_src,c = self.encoder(src, src_mask)
        #enc_src = [batch size, src len, hid dim]

        latent = self.latent(enc_src.view(-1,self.decoder.hid_dim,(self.maxlength))).view(-1,self.bottle_len*self.decoder.hid_dim) # Here the latent vector is generated
        if(use_noise):
            noise = 1*torch.randn_like(latent) # We add noise during training
            noise_latent = latent + noise
        else:
            noise_latent = latent
       
        noise_latent = torch.cat((noise_latent,c.view(-1,self.encoder.n_cat*self.decoder.hid_dim)),dim = 1).to(src.device) # This noised latent is concatenated with the 3 conditions
        
        enc_src = self.inverse(noise_latent).reshape(-1,self.decoder.hid_dim,self.bottle_len)
        enc_src = enc_src.permute(0, 2, 1)
        

        
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask = None)
    
      
        
        #output = [batch size, trg len, output dim]
        #attention = [batch size, n heads, trg len, src len]
        
        if(use_noise):
            return output, attention, latent
        else:
            return output, attention, latent