import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import tqdm
import numpy as np
from rdkit import Chem, RDLogger 
from rdkit.Chem import AllChem,Crippen
RDLogger.DisableLog('rdApp.*')  # RDKit is very annoying that it keeps printing out error

def is_SMILE(name):
  # Input a SMILE string and return if the string has valid grammar or not
  return(Chem.MolFromSmiles(name) is not None)

def gen_conformers(mol, numConfs=1, maxAttempts=1000, pruneRmsThresh=0.1, useExpTorsionAnglePrefs=True, useBasicKnowledge=True, enforceChirality=True, return_out = True):
    ids = AllChem.EmbedMultipleConfs(mol, numConfs=numConfs, maxAttempts=maxAttempts, pruneRmsThresh=pruneRmsThresh, useExpTorsionAnglePrefs=useExpTorsionAnglePrefs, useBasicKnowledge=useBasicKnowledge, enforceChirality=enforceChirality, numThreads=0)
    if(return_out):
        return(list(ids))
    
def check_logp(smiles):
    try:
        new_mol=Chem.MolFromSmiles(smiles)
        val = Crippen.MolLogP(new_mol)
        return val
    except:
        return None



def translate_sentence(src, model, device, smiledict,invdict, max_len = 220, randz = False, return_Z = False, C = None):
    # This function does the translation from a single source input to target
    maxlength = max_len
    model.eval()

    src_tensor = src.to(device).view(1,-1)
    
    src_mask = model.make_src_mask(src_tensor)
    
    with torch.no_grad():  
       
        enc_src,c = model.encoder(src_tensor, src_mask)
        if(C is not None): # If we have a condition
            C = torch.FloatTensor(C).view(-1,model.encoder.n_cat).to(device)
            conditions = []
            for i in range(model.encoder.n_cat):
                cat_embedding = torch.tensor([ (i + model.encoder.input_dim - model.encoder.n_cat) for _ in range(len(src))]).long().squeeze().to(device)
                conditions.append(C[:,i].view(-1,1) * model.encoder.tok_embedding(cat_embedding)) # Here we extract the embedding of the 3 categories
            c = torch.cat(conditions,dim = 1)
            

            c.to(device)
        
        z = model.latent(enc_src.view(-1,model.decoder.hid_dim,(maxlength))).view(-1,model.bottle_len*model.decoder.hid_dim)
        if(return_Z == True):
            return(z)
        if(randz == True):  # If we want to sample from the latent space P(Z,C) 
          z = torch.randn_like(z) 

        z = torch.cat((z,c.view(-1,model.encoder.n_cat*model.decoder.hid_dim)),dim = 1)# Concatenating the conditions

        enc_src = model.inverse(z).reshape(-1,model.decoder.hid_dim,model.bottle_len)

       
        enc_src = enc_src.permute(0, 2, 1)
      
       
    trg_indexes = [0] # 0 is the token for start

    for i in range(max_len):
        
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        
        trg_mask = model.make_trg_mask(trg_tensor)
        
        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask = None)
        
        pred_token = output.argmax(2)[:,-1].item()
        
        trg_indexes.append(pred_token)
        
        if pred_token == smiledict['?']:
            break
    output = ''
    for i in trg_indexes:
      output += invdict[i]
    

    return output[1:-1]

def get_valid_unique(generated_smiles, smilenames):
    # Return the list of valid unique smiles with their indeces from the list of smiles names
    valid_smiles = []
    valid_unique_smiles = []
    # Print the valid percentage
    indeces = []
    if(smilenames[0][0] == '!' and smilenames[0][-1] == '?'):
      smilenames = [s[1:-1] for s in smilenames]
      
    for mol in generated_smiles:
        if(is_SMILE(mol)):
           
            
            if(mol not in valid_unique_smiles):
                
                if(mol not in smilenames):
                    valid_unique_smiles.append(mol)
                    indeces.append(True)
                else:
                    indeces.append(False)
            
            else:
                indeces.append(False)
          
        else:
            indeces.append(False)

    assert len(valid_unique_smiles) != 0
    return(valid_unique_smiles,indeces)

def Batch_Beam_Search(model, device, smiledict, invidct, max_len = 220, randz = False, number = 100, C = None, beam_size = 2):
    pad_index = len(smiledict)
    # This function does the translation from a sources input to target

    model.eval()

    c = torch.zeros(number,model.encoder.n_cat*model.decoder.hid_dim).float().to(device)


    z = torch.randn(number,model.bottle_len*model.decoder.hid_dim).to(device)

    z = torch.cat((z,0*c),dim = 1)# Concatenating the conditions
    
    
    enc_src = model.inverse(z).reshape(-1,model.decoder.hid_dim,model.bottle_len)

    
    enc_src = enc_src.permute(0, 2, 1)
    beam_enc_src = torch.stack([enc_src for _ in range(beam_size)],dim = 1) # This is the encoder vector for encoder-decoder attention when treating batch*beam_size dimension as one unified dimension
    beam_enc_src = beam_enc_src.view(number*beam_size, -1, model.decoder.hid_dim)
    #trg_indexes = [0] # 0 is the token for start

    trg_indexes = []
    for _ in range(number):
        trg_indexes.append([[0] for _ in range(beam_size)])


    initial_targ = [[0] for _ in range(number)]

    prob_tracker = np.zeros((number,beam_size))


    trg_tensor = torch.LongTensor(initial_targ).to(device) # [number, 1]

    trg_mask = model.make_trg_mask(trg_tensor)

    with torch.no_grad():
        output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask = None)
    #output = [number, seq_len, len(dict)]

    normalized_probs = torch.softmax(output, dim = -1) # [number, seq_len, len(dict)]

    #print(normalized_probs.shape)
    top_k = []
    top_k_probs = []
    for n in range(number):
        top_k.append(torch.argsort(normalized_probs[n,-1,:])[-1*torch.arange(1,1+beam_size,1)])# Aranging from indeces that goes from highest probability to the lowest
        top_k_probs.append(normalized_probs[n,-1,:][top_k[-1]])


        
        
        
        
    for n in range(number):
        for k in range(beam_size):

            trg_indexes[n][k].append(top_k[n][k].item()) # keep track of the indeces


            prob_tracker[n][k] += np.log(top_k_probs[n][k].item()) # keep track of the probability

   
            
            
            
            
    for i in tqdm(range(max_len - 2)):
        k_probs = [[] for _ in range(number)]
        candidate_probs = [[] for _ in range(number)]
        candidate_seqs = [[] for _ in range(number)]
        completed_best = [0 for _ in range(number)]

        trg_tensor = torch.LongTensor(trg_indexes).view(number*beam_size,-1).to(device)

        trg_mask = model.make_trg_mask(trg_tensor)
        
        

        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, beam_enc_src, trg_mask, src_mask = None)
        output = output.view(number,beam_size,-1,len(invdict))


        for n in range(number):

            for k in range(beam_size):


                if(smiledict['?'] not in trg_indexes[n][k]): # If this sequence has not reached its end
                    k_prob = prob_tracker[n][k]


                    normalized_probs = torch.softmax(output[n][k], dim = -1) #[Seq_len, output dim]
                    k_prob_vec = k_prob + torch.log(normalized_probs[-1,:])
                    top_k = torch.argsort(k_prob_vec)[-1*torch.arange(1,1+beam_size,1)]

                    k_probs[n].append(k_prob_vec)

                    for j in range(beam_size):

                        temp_candidate = trg_indexes[n][k] + [top_k[j].item()]
                        candidate_seqs[n].append(temp_candidate)

                        candidate_probs[n].append(k_prob_vec[top_k[j]].item()) # This will append beam_size x beam_size number of candidate probabilities



                else:

                    #continue writing the fuction that does not alter the sequence
                    candidate_seqs[n].append((trg_indexes[n][k] + [pad_index])) # Padding and prepare for next iteration
                    candidate_probs[n].append(prob_tracker[n][k])
                    completed_best[n] += 1


                #print(np.array(candidate_probs).shape)







        # With top k selected from each of the k beams, choose the best k out of the k*k options
        normalized_candidate_probs = [[] for _ in range(number)]
        best_k_batch = []
        for n in range(number):
            for k in range(len(candidate_probs[n])):
                normalized_candidate_probs[n].append(
                candidate_probs[n][k]/(np.sum(np.array(candidate_seqs[n][k]) != pad_index)**0.5)
                )

            best_k = np.argsort(normalized_candidate_probs[n])[-1*np.arange(1,1+beam_size,1)]
            best_k_batch.append(best_k)


        candidate_seqs = candidate_seqs
        #print(len(candidate_probs) * len(candidate_probs[0]))
        #candidate_probs = np.array(candidate_probs)

        for n in range(number):
            for k in range(beam_size):
                trg_indexes[n][k] = candidate_seqs[n][best_k_batch[n][k]]
                prob_tracker[n][k] = candidate_probs[n][best_k_batch[n][k]]


    outputs = [[] for _ in range(number)]

    for n in range(number):
        for k in range(beam_size):
            output = ''
            for i in trg_indexes[n][k]:
                if(invdict[i] != '?'):
                    output += invdict[i]
                else:
                    break
            outputs[n].append(output[1:])
    return(outputs)





def Batch_Beam_Generate_Z(z, model, device, smiledict, invdict, max_len = 220, randz = False, C = None, beam_size = 2):
    # This function does the translation from a latent vector Z to target

    model.eval()
    pad_index = len(smiledict)
    
    number = len(z)
    c = torch.zeros(number,model.encoder.n_cat*model.decoder.hid_dim).float().to(device)
   
    
    z = torch.cat((z,0*c),dim = 1)# Concatenating the conditions
    
    
    enc_src = model.inverse(z).reshape(-1,model.decoder.hid_dim,model.bottle_len)

    
    enc_src = enc_src.permute(0, 2, 1)
    beam_enc_src = torch.stack([enc_src for _ in range(beam_size)],dim = 1) # This is the encoder vector for encoder-decoder attention when treating batch*beam_size dimension as one unified dimension
    beam_enc_src = beam_enc_src.view(number*beam_size, -1, model.decoder.hid_dim)
    #trg_indexes = [0] # 0 is the token for start

    trg_indexes = []
    for _ in range(number):
        trg_indexes.append([[0] for _ in range(beam_size)])


    initial_targ = [[0] for _ in range(number)]

    prob_tracker = np.zeros((number,beam_size))


    trg_tensor = torch.LongTensor(initial_targ).to(device) # [number, 1]

    trg_mask = model.make_trg_mask(trg_tensor)

    with torch.no_grad():
        output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask = None)
    #output = [number, seq_len, len(dict)]

    normalized_probs = torch.softmax(output, dim = -1) # [number, seq_len, len(dict)]

    #print(normalized_probs.shape)
    top_k = []
    top_k_probs = []
    for n in range(number):
        top_k.append(torch.argsort(normalized_probs[n,-1,:])[-1*torch.arange(1,1+beam_size,1)])# Aranging from indeces that goes from highest probability to the lowest
        top_k_probs.append(normalized_probs[n,-1,:][top_k[-1]])


        
        
        
        
    for n in range(number):
        for k in range(beam_size):

            trg_indexes[n][k].append(top_k[n][k].item()) # keep track of the indeces


            prob_tracker[n][k] += np.log(top_k_probs[n][k].item()) # keep track of the probability

   
            
            
            
            
    for i in tqdm(range(max_len - 2)):
        k_probs = [[] for _ in range(number)]
        candidate_probs = [[] for _ in range(number)]
        candidate_seqs = [[] for _ in range(number)]
        completed_best = [0 for _ in range(number)]

        trg_tensor = torch.LongTensor(trg_indexes).view(number*beam_size,-1).to(device)

        trg_mask = model.make_trg_mask(trg_tensor)
        
        

        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, beam_enc_src, trg_mask, src_mask = None)
        output = output.view(number,beam_size,-1,len(invdict))


        for n in range(number):

            for k in range(beam_size):


                if(smiledict['?'] not in trg_indexes[n][k]): # If this sequence has not reached its end
                    k_prob = prob_tracker[n][k]


                    normalized_probs = torch.softmax(output[n][k], dim = -1) #[Seq_len, output dim]
                    k_prob_vec = k_prob + torch.log(normalized_probs[-1,:])
                    top_k = torch.argsort(k_prob_vec)[-1*torch.arange(1,1+beam_size,1)]

                    k_probs[n].append(k_prob_vec)

                    for j in range(beam_size):

                        temp_candidate = trg_indexes[n][k] + [top_k[j].item()]
                        candidate_seqs[n].append(temp_candidate)

                        candidate_probs[n].append(k_prob_vec[top_k[j]].item()) # This will append beam_size x beam_size number of candidate probabilities



                else:

                    #continue writing the fuction that does not alter the sequence
                    candidate_seqs[n].append((trg_indexes[n][k] + [pad_index])) # Padding and prepare for next iteration
                    candidate_probs[n].append(prob_tracker[n][k])
                    completed_best[n] += 1


                #print(np.array(candidate_probs).shape)







        # With top k selected from each of the k beams, choose the best k out of the k*k options
        normalized_candidate_probs = [[] for _ in range(number)]
        best_k_batch = []
        for n in range(number):
            for k in range(len(candidate_probs[n])):
                #print(candidate_seqs[n][k])
                #print(np.sum(np.array(candidate_seqs[n][k]) != pad_index)**2)
                normalized_candidate_probs[n].append(
                
                candidate_probs[n][k]/(np.sum(np.array(candidate_seqs[n][k]) != pad_index)**1)
                    
                )

            best_k = np.argsort(normalized_candidate_probs[n])[-1*np.arange(1,1+beam_size,1)]
            best_k_batch.append(best_k)


        candidate_seqs = candidate_seqs
       
        for n in range(number):
            for k in range(beam_size):
                trg_indexes[n][k] = candidate_seqs[n][best_k_batch[n][k]]
                prob_tracker[n][k] = candidate_probs[n][best_k_batch[n][k]]


    outputs = [[] for _ in range(number)]

    for n in range(number):
        for k in range(beam_size):
            output = ''
            for i in trg_indexes[n][k]:
                if(invdict[i] != '?'):
                    output += invdict[i]
                else:
                    break
            outputs[n].append(output[1:])

    return(outputs)
