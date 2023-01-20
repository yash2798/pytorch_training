#!/usr/bin/env python
# coding: utf-8

# In[8]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[4]:


import torch


# In[5]:


torch.__version__


# In[9]:


#1-d tensor
a=torch.tensor([1,2,3,4,5])
a.dtype
a.type()


# In[11]:


b=torch.FloatTensor([1,2,3,4])
b.dtype
b.type()


# In[12]:


a.size()
b.size()


# In[13]:


a.ndimension()
b.ndimension()


# In[17]:


a_col=a.view(5,1)
a_col
a_col.dtype
a_col.ndimension()


# In[22]:


b_new=b.view(-1,4)
b_new
b_new.ndimension()


# In[29]:


import numpy as np
c=np.arange(10)[1:10]
c=c.reshape(3,3)
c_tensor=torch.from_numpy(c)
c_tensor
c_tensor.ndimension()


# In[33]:


import pandas as pd
s=pd.Series([0,1,2,3,4])
s
s_tensor=torch.tensor(s.values)
s_tensor
s_tensor.ndimension()


# In[35]:


ls=s_tensor.tolist()
type(ls)


# In[36]:


#representing vectors using 1d arrays
u=torch.tensor([1,0])
v=torch.tensor([0,1])
2*u+3*v #LC of vectors


# In[37]:


h=torch.tensor([1,6,-1,2])
h+1 #just like numpy broadcasting


# In[39]:


d=torch.linspace(0,1,steps=25)
d


# In[41]:


x=torch.linspace(0,2*np.pi,steps=100)
y=torch.sin(x)
import matplotlib.pyplot as plt
plt.plot(x.numpy(),y.numpy())


# In[ ]:




