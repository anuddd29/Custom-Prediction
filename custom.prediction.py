#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('gsutil -m cp gs://anuddd29/model/cats_vs_dogs.h5 . ')


# In[7]:


import tensorflow as tf

model =tf.keras.models.load_model('cats_vs_dogs.h5')
model.summary()


# In[10]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np


# In[17]:


classes = ['Cat', 'Dog']

def display_pred(image_path):
    plt.imshow(plt.imread(image_path))

    x = tf.keras.preprocessing.image.load_img(image_path, target_size= (128,128))
    x = tf.keras.preprocessing.image.img_to_array(x)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x= np.expand_dims (x, axis=0)

    pred = model.predict(x)[0]
    plt.title('Pred: {}'.format(classes[int(pred > 0.5)]))
    plt.show()

    print(pred)


# In[18]:


images = ['images/cat1.jpg', 'images/cat2.jpg',
        'images/dog1.jpg','images/dog2.jpg'] 
display_pred(images[0])
        


# In[19]:


display_pred(images[-1])


# In[22]:


get_ipython().run_cell_magic('writefile', 'prediction.py', "\nimport tenserflow as tf\nimport numpy as np\nimport os\nimport base64\n\nMODEL_NAME = 'cats_vs_dogs.h5'\nCLASS_NAMES = ['Cat', 'Dog']\n\nclass CatsvsDogsPrediction\n    def __init__(self, model):\n        self.model = model\n        \n    def _preprocess(self, instances, size=128):\n        num_examples = len(instances)\n        x_batch = np.zeros((num_examples, size,size, 3))\n        for i in range(num_examples):\n        x = np.array(bytearray(base64.b64decode(instance[i])))\n        x = np.reshape(x, (size, size, 3))\n        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)\n        x = batch[i] = x\n        return x_batch\n    \n    def _postprocess(self, preds):\n        results = []\n        for i, pred in enumerate(preds):\n            p = np.squeeze(pred)\n            results.append({\n                'index': i,\n                'class_name': CLASS_NAMES[int(p> 0.5)],\n                'raw_value': '{:.4f}'.format(p)\n            })\n            \n    def predict(self, instance, **kwargs):\n        if 'size' in kwargs:\n            size = int(kwargs.get('size'))\n        else:\n            size = 128\n        # Preprocess\n        x_batch = self._preprocess(instances, size)\n        \n        # Predict\n        preds = self._model.predict(x_batch)\n        \n        # Post process\n        results = self._postprocess(preds)\n        return results\n        \n        \n        return\n        \n    @classmethod\n    def from_path(cls, model_dir):\n        model = tf.keras.models.load_model(os.path.join(model_dir, MODEL_NAME))\n        return cls(model)")


# In[23]:


get_ipython().run_cell_magic('writefile', 'setup.py', "from setuptools import setup\n\nsetup(\n    name= 'cats_vs_dogs',\n    version = '0.0.1',\n    include_package_data= False,\n    scripts = ['prediction.py']\n)")


# In[25]:


get_ipython().system('python3 setup.py sdist --formats=gztar')


# In[29]:


get_ipython().system('gsutil cp dist/cats_vs_dogs-0.0.1.tar.gz gs://anuddd29/dist/')


# In[ ]:





# In[31]:


from googleapiclient import discovery
from PIL import Image 
import os
import base64


# In[33]:


service = discovery.build('ml', 'v1', cache_discovery= False)


# In[66]:


def get_pred_from_model(body,project_name, model_name):
    service.projects().predict(
        name ='projects/{}/models/{}'.format(project_name, model_name),
        body=body
    )


# In[63]:


project_name = 'custom-prediction-292600'
model_name = 'cats_vs_dogs'


# In[57]:


images


# In[58]:


instances = []
size = 128

for image in images:
    img = Image.open(image)
    img = img.resize((size, size), Image.ANTIALIAS)
    
    instances.append(
        base64.b64encode(img.tobytes()).decode()
    )
    img.close()


# In[59]:


body = {
    'instances': instances,
    'size': size
}


# In[69]:


response = get_pred_from_model(body, project_name, model_name)


# In[72]:


response


# In[ ]:





# In[ ]:





# In[ ]:




