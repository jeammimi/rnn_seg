
# coding: utf-8

# In[3]:


import sys


sys.path.append("../../")
from src.models.train_model import generator


sample = generator(size_sample=10, type="BDSD", ndim=2, sub=False, traj=False, old=True)

i = 0
for s in sample:
    i += 1
    print(s)
    break


# In[ ]:


sample.next()


# In[ ]:
