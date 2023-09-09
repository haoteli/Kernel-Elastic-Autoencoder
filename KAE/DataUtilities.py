import os
import pandas as pd
import numpy as np
import math
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
      
      
      
def prepare_data(smiles, conditions, load_from_pickle_directory = None, token_threshold = 200):
  # smiles: array of strings
  # conditions: array of floats
  # load_from_pickle_directory: directory of the pickle file to load batch,batch_test,smiledict,invdict and csv variables.
  


  if load_from_pickle_directory is not None:

      batch,batch_test,smiledict,invdict,csv = pickle.load(open(load_from_pickle_directory,'rb'))
      smilenames = [smile for smile in csv]

      print('Read Checkpoint File Complete.')
      print()
      return(batch,batch_test,smiledict,invdict,csv,smilenames)



  else:

      smiles = smiles
      properties = conditions

      sub_data = []
      feats = properties
      combined_data = smiles
      not_cononical = 0
      for s in combined_data:
          sub_data.append(s)
          
      sub_data = np.array(sub_data)
      csv = np.array(sub_data)

      thresh = token_threshold # We will be confining the entries to be less than this many numbers of characters

      ok_ind = []
      for s in csv:
          if(len(s) <= thresh):
              ok_ind.append(True)
          else:
              ok_ind.append(False)



      print('Percentage below allowed threshold',np.sum(ok_ind)/len(ok_ind))
      
      feats = feats[ok_ind]
      csv = csv[ok_ind]
      properties = feats # Getting all the properties [N, number of categories]
      n_cat = properties.shape[-1] if (len(properties.shape) == 2) else 1

      if(n_cat == 1):
          properties = properties.reshape((-1,1))

      normalized_properties = feats#np.zeros_like(properties)

      print('Number of Samples:',len(csv))

      data = csv
      for i in range(len(data)):
        data[i] = '!' + data[i][:] + '?' # This adds a start and end symbol for data
      data = [[char for char in seq] for seq in data] # This makes data seperate by characters

      # Make the conversion and inverse conversion dictionaries
      smiledict = {}
      label= 0
      invdict = {}
      for smile in data:
        for char in smile:
          if char not in smiledict.keys():
            smiledict[char] = label
            invdict[label] = char
            label += 1

      pad_index = len(smiledict) # This is the token for the pad
      invdict[pad_index] = ' ' # Make the padding an empty character

      maxlength = 0 # Define a maximum length for all the data
      minlength = 1000 # Define a minimum length for all the data
      for seq in data:
        if len(seq)>maxlength:
          maxlength = len(seq)
        if len(seq) < minlength:
          minlength = len(seq)
      
      maxlength = maxlength + 4 # Allowing more space in case needed

      print('The maximum length among the data set is '+str(maxlength))
      print('The minimum length among the data set is '+str(minlength))

      
      
      def char2ind(seq,trgdict = smiledict): # This translate a character to an index
        return([trgdict[char] for char in seq])

      def ind2seq(seq):
        _ = ''
        for i in range(len(seq)):
          _ = _+invdict[seq[i].item()]
        return(_)


      def pad(seq,maxlength): # Pad a sequence to a given length
        maxlabel = len(smiledict)
        for i in range(maxlength - len(seq)):
          seq.append(maxlabel)
          

      testsize = math.ceil(len(data)/10) # Here we use 90/100 of the dataset for training

      train_data = data[:-testsize]
      test_data = data[-testsize:]

      train_properties = normalized_properties[:-testsize]
      test_properties = normalized_properties[-testsize:]





      batch_size = 32
      

      for i in range(len(train_data)): # Change all characters to index
        train_data[i] = char2ind(train_data[i])


      for i in range(len(test_data)): # Similarly prepare the test data
        test_data[i] = char2ind(test_data[i])


      batch = [] # Define the batch
      batch_test = [] # Define the batch for test

     

      for i in range(0,len(train_data)-batch_size,batch_size):
        # For every batch do the following
        _batch_maxlength = maxlength # Use absolute longest length


        for j in range(batch_size):
          pad(train_data[i+j],_batch_maxlength - n_cat) # Pad all the sequences in the batch to the maximum length - number of categories

        seq = torch.tensor(train_data[i:i+batch_size]).float()
        temp = torch.cat((seq, torch.FloatTensor(normalized_properties[i:i+batch_size]).view(batch_size,-1)),dim = 1) # Concatenating the categories at the end of each string after padding
        batch.append(temp) # Add the batch data to the list

      for i in range(len(test_data)):
        pad(test_data[i],maxlength - n_cat) # Pad all test data to the max length 

      batch_test.append( torch.cat((torch.FloatTensor(test_data[:]), torch.FloatTensor(test_properties[:]).view(-1,n_cat) ),dim = 1) )

      smilenames = [smile for smile in csv]

      pickle.dump((batch,batch_test,smiledict,invdict,csv), open('cpth_data.pkl', 'wb'))
      print('Dumped Data to cpth_data.pkl')
      
      return(batch,batch_test,smiledict,invdict,csv,smilenames)
      
      

class CustomDataset(Dataset):

    def __init__(self, batch, maxlength):
        # Initialize data, download, etc.
        # read with numpy or pandas



        # here the first column is the class label, the rest are the features
        self.x_data = torch.stack(batch).view(-1,maxlength) # size [n_samples, n_features]
        self.y_data = torch.stack(batch).view(-1,maxlength) # size [n_samples, 1]

        self.n_samples = len(self.x_data)
    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples
