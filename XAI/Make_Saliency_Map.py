import torch
from torch.utils.data import Dataset, Dataloader
import torch.nn as nn

import matplotlib.pyplot as splt
import seaborn as sns



'''
Add code for model here
'''

'''
Add code for dataloader here
'''

ims, labels = next(iter(test_loader))
ims.requires_grad_(True)
out = model.forward(ims)
classes = [x.argmax(dim = 0) for x in out]
model.zero_grad()

#Secoond forward pass required to receiev the gradients
out = model.forward(ims) 

#Tensor with shape of output for first image
one_hot_output = torch.zeros(out.size())

#Set predcited class in one_hot_out to 1 for each image in order to get gradients in model with respect to that class and get predicted classes
classes = []
for i in range(len(one_hot_output)):
  one_hot_output[i][out[i].argmax()] = 1
  classes.append(out[i].detach().argmax())
  
#Calculate gradient of model with respect to predicted class
out.backward(gradient=one_hot_output, retain_graph=True) 

#Get maximum gradient over channesls for saliency map
saliency, _ = torch.max(ims.grad.data.abs(), dim=1) 
  
  
#Plot results for each image
for i in range(len(one_hot_output)):
  fig, axes = plt.subplots(2, 1, figsize=(15, 10))
  img = ims[i].detach().numpy()
  
  axes[0].imshow(img.reshape(128, 256), cmap = 'gray')
  axes[0].set_title('Original Image')
  axes[0].axis('off')
  
  axes[1].imshow(saliency[i].detach().numpy(), cmap='hot')
  axes[1].set_title('Saliency Map')
  axes[1].axis('off')
  
  plt.tight_layout()
  fig.suptitle(f'Image of class {labels.numpy()[i]} and saliency map - model predicted {classes[i]}', y = 1.05, size = 30)
  plt.show()
