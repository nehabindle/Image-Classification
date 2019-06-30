
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from skimage.feature import hog 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.ensemble import RandomForestClassifier


# Defining Path for the image and Labels folder

# In[2]:


#Train Image and Labels
X_train_path = 'traffic/traffic/train'
y_train_path = 'traffic/traffic/train.labels'

#Test Image and Labels
X_test_path = 'traffic/traffic/test'
y_test_path = 'traffic/traffic-small/test.labels'


# Function to load folder into arrays and then it returns that same array

# In[3]:


def loadImages(path):
    # Put files into lists and return them as one list of size 4
    image_files = sorted([os.path.join(path, file)
         for file in os.listdir(path) if file.endswith('.jpg')])
 
    return image_files


# Function to read labels and store in into a dataframe

# In[4]:


'''def read_labels(file_path):
    y_train = []
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip()
            y_train.append(line)
    return y_train'''

y_train = np.loadtxt(y_train_path)
y_test = np.loadtxt(y_test_path)

y_labels = pd.DataFrame(y_train)
y_labels[0].value_counts()
print(type(y_train[0]))


# In this step in order to visualize the change, we are going to create two functions to display the images the first being a one to display one image and the second for two images

# In[5]:


# Display one image
def display_one(a, title1 = "Original"):
    plt.imshow(a), plt.title(title1)
    plt.xticks([]), plt.yticks([])
    plt.show()
# Display two images
def display(a, b, title1 = "Original", title2 = "Edited"):
    plt.subplot(121), plt.imshow(a), plt.title(title1)
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(b), plt.title(title2)
    plt.xticks([]), plt.yticks([])
    plt.show()


# Establish a base size for all images fed into the model

# In[6]:


# Preprocessing
def processing(data):
    # loading image
    img = [cv2.imread(i, cv2.IMREAD_UNCHANGED) for i in data[:]]
    print('Original size',img[1].shape)
    # --------------------------------
    # setting dim of the resize
    height = 50
    width = 50
    dim = (width, height)
    res_img = []
    for i in range(len(img)):
        res = cv2.resize(img[i], dim, interpolation=cv2.INTER_LINEAR)
        res_img.append(res)
    
    return res_img
# # Checcking the size
#     print("RESIZED", res_img[1].shape)
    
#     # Visualizing one of the images in the array
#     original = res_img[1]
#     display_one(original)


# In[7]:


X_train = loadImages(X_train_path)
X_train_P = processing(X_train)


# In[8]:


# print(X_train_P)
X_train_new = np.array(X_train_P)
print(X_train_new.shape)


# In[9]:


X_test = loadImages(X_test_path)
X_test_P = processing(X_test)


# In[10]:


X_test_new = np.array(X_test_P)
print(X_test_new.shape)


# In[11]:


from skimage import data, exposure


# In[12]:


def test_hog(img_no):
    hog_feature, hog_img = hog(X_train_new[img_no], orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualize=True, multichannel=True)

    display(X_train_new[img_no], hog_img, title2="Histogram of gradients")
    print("hog_feature vector: ", hog_feature.shape, type(hog_feature))
    print("hog_image vector: ", hog_img.shape, type(hog_img))
    
test_hog(50)

def calc_hog(img): # hog_feature_vectors
    return hog(img, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1), multichannel=True)

X_train_hog = np.array( [calc_hog(img) for img in X_train_new] )
X_test_hog = np.array( [calc_hog(img) for img in X_test_new] )


# In[13]:


print(X_train_hog.shape)
print(X_test_hog.shape)


# In[16]:


# from sklearn.decomposition import PCA
# pca = PCA(n_components=30, svd_solver='full')
# X_pca = pca.fit(X_train_hog)     


# In[14]:


from sklearn import svm


# In[15]:


svm_clf = svm.SVC(decision_function_shape='ovo', kernel = 'rbf',degree = 3 )
svm_clf.fit(X_train_hog, y_train)


# In[18]:


y_predict=svm_clf.predict(X_test_hog)


# In[16]:


# from sklearn.metrics import accuracy_score,f1_score
# f1_score(y_predict,y_labels,average="weighted")


# In[19]:


np.savetxt('test.txt2',y_predict,fmt='%d',delimiter="\n")


# In[22]:


# r_clf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=0,criterion='gini')


# In[23]:


# r_clf.fit(X_train_hog, y_train)


# In[25]:


# y_predict1=r_clf.predict(X_test_hog)


# In[21]:


#f1_score(y_predict1,y_labels,average="weighted")


# In[26]:


# np.savetxt('test1.txt',y_predict1,fmt='%d',delimiter="\n")

