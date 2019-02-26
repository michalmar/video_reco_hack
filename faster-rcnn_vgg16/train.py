from utils_train import get_data, get_anchor_gt, display_image, model_init, train, display_after_training, get_img_output_length
from utils_train import Config
import os
import time
import random
import pprint
import pickle
import numpy as np


# ### Start training

# In[16]:


base_path = '.'

# train_path =  './Dataset/OpenImages/annotation.txt' # Training data (annotation file)
train_path =  './Dataset/airport/train_annotation.txt' # Training data (annotation file)

num_rois = 4 # Number of RoIs to process at once.

# Augmentation flag
horizontal_flips = True # Augment with horizontal flips in training. 
vertical_flips = True   # Augment with vertical flips in training. 
rot_90 = True           # Augment with 90 degree rotations in training. 

output_weight_path = os.path.join(base_path, 'model/model_frcnn_vgg.hdf5')

record_path = os.path.join(base_path, 'model/record.csv') # Record data (used to save the losses, classification accuracy and mean average precision)

base_weight_path = os.path.join(base_path, 'model/vgg16_weights_tf_dim_ordering_tf_kernels.h5')

config_output_filename = os.path.join(base_path, 'model_vgg_config.pickle')


# In[17]:


# Create the config
C = Config()

C.use_horizontal_flips = horizontal_flips
C.use_vertical_flips = vertical_flips
C.rot_90 = rot_90

C.record_path = record_path
C.model_path = output_weight_path
C.num_rois = num_rois

C.base_net_weights = base_weight_path


# In[18]:


#--------------------------------------------------------#
# This step will spend some time to load the data        #
#--------------------------------------------------------#
st = time.time()
train_imgs, C.classes_count, C.class_mapping = get_data(train_path)
print()
print('Spend %0.2f mins to load the data' % ((time.time()-st)/60) )


# In[19]:


if 'bg' not in C.classes_count:
	C.classes_count['bg'] = 0
	C.class_mapping['bg'] = len(C.class_mapping)
# e.g.
#    classes_count: {'Car': 2383, 'Mobile phone': 1108, 'Person': 3745, 'bg': 0}
#    class_mapping: {'Person': 0, 'Car': 1, 'Mobile phone': 2, 'bg': 3}
C.class_mapping = C.class_mapping

print('Training images per class:')
pprint.pprint(C.classes_count)
print('Num classes (including bg) = {}'.format(len(C.classes_count)))
print(C.class_mapping)

# Save the configuration
with open(config_output_filename, 'wb') as config_f:
	pickle.dump(C,config_f)
	print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(config_output_filename))


# In[20]:


# Shuffle the images with seed
random.seed(1)
random.shuffle(train_imgs)

print('Num train samples (images) {}'.format(len(train_imgs)))


# In[21]:


# Get train data generator which generate X, Y, image_data
data_gen_train = get_anchor_gt(train_imgs, C, get_img_output_length, mode='train')


# #### Explore 'data_gen_train'
# 
# data_gen_train is an **generator**, so we get the data by calling **next(data_gen_train)**

# In[22]:


X, Y, image_data, debug_img, debug_num_pos = next(data_gen_train)

# display_image(X, Y, image_data, debug_img, debug_num_pos, C)


# #### Build the model

# In[24]:

(C.model_all, C.model_rpn, C.model_classifier, record_df) = model_init(C)
# In[27]:


# Training setting
C.total_epochs = len(record_df)
C.r_epochs = len(record_df)

C.epoch_length = 227
C.num_epochs = 20
C.iter_num = 0

C.total_epochs += C.num_epochs

C.losses = np.zeros((C.epoch_length, 5))
C.rpn_accuracy_rpn_monitor = []
C.rpn_accuracy_for_epoch = []

if len(record_df)==0:
    C.best_loss = np.Inf
else:
    C.best_loss = np.min(r_curr_loss)


# In[28]:


print(len(record_df))


# In[ ]:

record_df = train(C, data_gen_train, record_df)

# In[ ]:
display_after_training(C, record_df)