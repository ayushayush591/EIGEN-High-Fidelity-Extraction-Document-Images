import matplotlib.pyplot as plt
import os
import json
import pandas as pd
import numpy as np
def normalize_bbox(bbox, width, height):
     return [
         int(1000 * (bbox[0] / width)),
         int(1000 * (bbox[1] / height)),
         int(1000 * (bbox[2] / width)),
         int(1000 * (bbox[3] / height)),
     ]
def CORD_train(directory):
  ####################
  # cord Dataset
  ####################

  # Process the Raw Labelled cord Data
  # Data consisting of 112 annotated medical documents
#   directory = 'CORD/train/json'
  
  # iterate over files
  words = []
  bbox = []
  labels = []
  for filename in sorted(os.listdir(directory)):
      f = os.path.join(directory, filename)
      # checking if it is a file
      if os.path.isfile(f):
        dataraw = open(f)
        jdata = json.load(dataraw)
        height = jdata['meta']['image_size']['height']
        width = jdata['meta']['image_size']['width']
        word = []
        boxes = []
        label = []
        for feature in jdata['valid_line']:
          for ele in feature['words']:
              txt = ele['text']

              # bounding box - (upper left, lower right) upper left is (x1, y3)
              # and the lower right is (x3, y1)
              x1 = ele['quad']['x1']
              y1 = ele['quad']['y1']
              x3 = ele['quad']['x3']
              y3 = ele['quad']['y3']
              
              box = [x1, y1, x3, y3]
              box = normalize_bbox(box, width=width, height=height) 

              if len(txt) < 1: 
                continue
              if min(box) < 0 or max(box) > 1000: # another bug in which a box had -4
                continue
              if ((box[3] - box[1]) < 0) or ((box[2] - box[0]) < 0): # another bug in which a box difference was -12
                continue
              # ADDED

              word.append(txt)
              boxes.append(box) 
              label.append(feature['category'])
        words.append(word) 
        bbox.append(boxes) 
        labels.append(label)
  return words,labels,bbox 
token_train, label_train, boxe_train = CORD_train("CORD/train/json")
token_test, label_test, boxe_test= CORD_train("CORD/test/json")
token_dev, label_dev, boxe_dev = CORD_train("CORD/dev/json")
def preprossing(label):
    for i in range(len(label)):
        for j in range(len(label[i])):
            if (label[i][j] == "menu.nm") | (label[i][j] == 'menu.nm') | (label[i][j] == 'menu.sub_nm') | (label[i][j] == 'void_menu.nm') | (label[i][j] == 'menu.vatyn'):
                label[i][j] = 0
            elif (label[i][j] == 'menu.cnt') | (label[i][j] == 'total.menuqty_cnt') | (label[i][j] == 'menu.sub_cnt') | (label[i][j] == 'total.menutype_cnt') | (label[i][j] == 'menu.num'):
                label[i][j] = 2   
            else:
                label[i][j] = 1
    return label                
label_train=preprossing(label_train)
label_test=preprossing(label_test)
label_dev=preprossing(label_dev)

def cage_data(word,label,box):
    words=[item for sublist in word for item in sublist]
    bbox=[item for sublist in box for item in sublist]
    labels=[item for sublist in label for item in sublist]
    return words,labels,bbox
lf=8
# dic={242:4,1297:20,2587:40}
# l_ocr = ocr_train[:lf]
# u_ocr = ocr_train[lf:]
token_train_lab=token_train[:lf]
label_train_lab=label_train[:lf]
boxe_train_lab=boxe_train[:lf]
token_train_unl=token_train[lf:]
label_train_unl=label_train[lf:]
boxe_train_unl=boxe_train[lf:]

cage_token_train_lab, cage_label_train_lab, cage_boxe_train_lab=cage_data(token_train_lab, label_train_lab, boxe_train_lab)
cage_token_train_unl, cage_label_train_unl, cage_boxe_train_unl=cage_data(token_train_unl, label_train_unl, boxe_train_unl)
cage_token_test, cage_label_test, cage_boxe_test=cage_data(token_test, label_test, boxe_test)
cage_token_dev, cage_label_dev, cage_boxe_dev=cage_data(token_dev, label_dev, boxe_dev)

import enum
# Create Label Class
class ClassLabels(enum.Enum):
    # menu.cnt = 1
    # menu.discountprice = 2
    # menu.etc = 3
    # menu.itemsubtotal = 4
    # menu.nm = 5
    # menu.num = 6
    # menu.price = 7
    # menu.sub_cnt = 8
    # menu.sub_etc = 9
    # menu.sub_nm = 10
    # menu.sub_price = 11 
    # menu.sub_unitprice = 12
    # menu.unitprice = 13
    # menu.vatyn = 14
    # sub_total.discount_price = 15
    # sub_total.etc = 16
    # sub_total.othersvc_price = 17
    # sub_total.service_price = 18
    # sub_total.subtotal_price = 19
    # sub_total.tax_price = 20
    # total.cashprice = 21
    # total.changeprice = 22
    # total.creditcardprice = 23
    # total.emoneyprice = 24
    # total.menuqty_cnt = 25
    # total.menutype_cnt = 26
    # total.total_etc = 27
    # total.total_price = 28
    # void_menu.nm = 29
    # void_menu.price = 30
    MENU = 0
    PRICE = 1
    # CASH = 3
    QUANTITY = 2
  

THRESHOLD = 0.8

texttrigger = {"total","tax","price","discount","non","change","subtotal","%",",",".00",".000",".0",".0000","Kembali","cash","card","amount","tendered","net","sales",}
fieldtrigger = {"qty","1.00xITEMS","x","X","(Qty"}
fieldtriggerz = {"qty","item","items","x"}


from spear4HighFidelity.spear.labeling import labeling_function, ABSTAIN, preprocessor, continuous_scorer
import re

@preprocessor()
def convert_to_lower(x):
    return x.lower().strip()
    
@labeling_function(resources=dict(keywords=texttrigger),pre=[convert_to_lower],label=ClassLabels.PRICE)
def LF1(x,**kwargs):    
    if (len(kwargs["keywords"].intersection(x.split())) > 0):
        return ClassLabels.PRICE
    else:
        return ABSTAIN

@labeling_function(resources=dict(keywords=fieldtrigger),pre=[convert_to_lower],label=ClassLabels.QUANTITY)
def LF2(x,**kwargs):
    if (len(kwargs["keywords"].intersection(x.split())) > 0):
        return ClassLabels.QUANTITY
    else:
        return ABSTAIN

@labeling_function(resources=dict(keywords=fieldtriggerz),pre=[convert_to_lower],label=ClassLabels.QUANTITY)
def LF3(x,**kwargs):
    for pattern in kwargs["keywords"]:    
        if re.search(pattern,x, flags= re.I):
            return ClassLabels.QUANTITY
    return ABSTAIN

@labeling_function(pre=[convert_to_lower],label=ClassLabels.PRICE)
def LF4(x):
    if x.isnumeric() and int(x)>=10:
        return ClassLabels.PRICE
    else:
        return ABSTAIN

@labeling_function(pre=[convert_to_lower],label=ClassLabels.QUANTITY)
def LF5(x):
    if x.isnumeric() and int(x)<5:
        return ClassLabels.QUANTITY
    else:
        return ABSTAIN        

@labeling_function(resources=dict(keywords=texttrigger),pre=[convert_to_lower],label=ClassLabels.PRICE)
def LF6(x,**kwargs):
    for pattern in kwargs["keywords"]:    
        if re.search(pattern,x, flags= re.I):
            return ClassLabels.PRICE
    return ABSTAIN

#  and LF1(x)==ABSTAIN and LF2(x)==ABSTAIN and LF3(x)==ABSTAIN and LF4(x)==ABSTAIN and LF5(x)==ABSTAIN and LF6(x)==ABSTAIN     

@labeling_function(pre=[convert_to_lower],label=ClassLabels.MENU)
def LF7(x,**kwargs):    
    if x.isalnum() and LF1(x)==(None, None)  and LF5(x)==(None, None):
        return ClassLabels.MENU
    else:
        return ABSTAIN
    
from spear4HighFidelity.spear.labeling import LFSet

LFS = [LF1,
       LF2,
       LF3,
       LF4,
       LF5,
       LF6,
       LF7
      ]

rules = LFSet("CORD_LF")
rules.add_lf_list(LFS)

from spear4HighFidelity.spear.labeling import PreLabelsWithContext
import numpy as np

X = np.array(cage_token_train_unl)
bbox = np.array(cage_boxe_train_unl)
# bbox
Y = np.array(cage_label_train_unl)
# print(Y)

R = np.zeros((X.shape[0],len(rules.get_lfs())))

##############################
# Modified Prelabels
##############################
cord_noisy_labels = PreLabelsWithContext(name="cord",
                            data=X,
                            # data_feats=bbox,
                            gold_labels=Y,
                            rules=rules,
                            labels_enum=ClassLabels,num_classes=3)

L,S = cord_noisy_labels.get_labels()

#234 -> 1%     
#311 -> 1%
#1012 -> 5%
#1994 -> 10%
# Preprocessing
labelling_fac=1994
X_T = np.array(cage_token_train_lab)
Y_T = np.array(cage_label_train_lab) 
X_U = np.array(cage_token_train_unl)
Y_U = np.array(cage_label_train_unl)
bbox_l = np.array(cage_boxe_train_lab)
test_size = len(X_T)
U_size = len(X_U)
bbox_U = np.array(cage_boxe_train_unl)
n_lfs = len(rules.get_lfs())


# print(n_lfs)

# Paths
path_json = 'Paths/cord_json_1%.json'
T_path_pkl = 'Paths/cord_pickle_T_1%.pkl' #test data - have true labels
U_path_pkl = 'Paths/cord_pickle_U_1%.pkl' #unlabelled data - don't have true labels

log_path_cage_1 = 'Paths/cord_log_1_1%.txt' #cage is an algorithm, can be found below
params_path = 'Paths/cord_params_1%.pkl' #file path to store parameters of Cage, used below

# Generate Noisy labels
from spear4HighFidelity.spear.labeling import PreLabels

cord_noisy_labels_lab = PreLabels(name="cord",
                               data=X_T,
                               gold_labels=Y_T,
                               data_feats=bbox_l,
                               rules=rules,
                               labels_enum=ClassLabels,
                               num_classes=3)
L_lab,S_lab = cord_noisy_labels_lab.get_labels()                               
cord_noisy_labels_lab.generate_pickle(T_path_pkl)

cord_noisy_labels.generate_json(path_json) #generating json files once is enough

cord_noisy_labels_unl = PreLabels(name="cord",
                               data=X_U,
                               data_feats=bbox_U,
                               rules=rules,
                               labels_enum=ClassLabels,
                               num_classes=3) #note that we don't pass gold_labels here, for the unlabelled data
L_unl,S_unl = cord_noisy_labels_unl.get_labels()
cord_noisy_labels_unl.generate_pickle(U_path_pkl)

X_TI = np.array(cage_token_test)
Y_TI = np.array(cage_label_test)
X_D = np.array(cage_token_dev)
Y_D = np.array(cage_label_dev)
bbox_TI = np.array(cage_boxe_test) 
bbox_D = np.array(cage_boxe_dev) 
test_size = len(X_TI)
print(Y_TI.shape)
Z_size = len(X_D)
# n_lfs = len(rules.get_lfs())

# print(n_lfs)

# Paths
path_json = 'Paths/cord_json.json'
TI_path_pkl = 'Paths/cord_test_pickle_T_1%.pkl' #test data - have true labels
Z_path_pkl = 'Paths/cord_dev_pickle_U_1%.pkl' #unlabelled data - don't have true labels

log_path_cage_1 = 'Cages_Paths/cord_log_1.txt' #cage is an algorithm, can be found below
params_path = 'Cages_Paths/cord_params.pkl' #file path to store parameters of Cage, used below


# Generate Noisy labels
from spear4HighFidelity.spear.labeling import PreLabels

cord_noisy_labels_test = PreLabels(name="cord",
                               data=X_TI,
                               gold_labels=Y_TI,
                               data_feats=bbox_TI,
                               rules=rules,
                               labels_enum=ClassLabels,
                               num_classes=3)
L_test,S_test = cord_noisy_labels_test.get_labels()                               
cord_noisy_labels_test.generate_pickle(TI_path_pkl)

cord_noisy_labels.generate_json(path_json) #generating json files once is enough

cord_noisy_labels_val = PreLabels(name="cord",
                               data=X_D,
                               gold_labels=Y_D,
                               data_feats=bbox_D,
                               rules=rules,
                               labels_enum=ClassLabels,
                               num_classes=3) #note that we don't pass gold_labels here, for the unlabelled data
cord_noisy_labels_val.generate_pickle(Z_path_pkl)
L_val,S_val = cord_noisy_labels_val.get_labels()

from spear4HighFidelity.spear.cage import Cage
cage = Cage(path_json = path_json, n_lfs = n_lfs)

probs = cage.fit_and_predict_proba(path_pkl = U_path_pkl, path_test = T_path_pkl, path_log = log_path_cage_1, \
                                   qt = 0.9, qc = np.array([0.85]), metric_avg = ['micro'], n_epochs = 200, lr = 0.01)
labels = np.argmax(probs, 1)
print("probs shape: ", probs.shape)
print("labels shape: ",labels.shape)

from spear4HighFidelity.spear.labeling import PreLabels
import numpy as np

R = np.zeros((X.shape[0],len(rules.get_lfs())))

cord_noisy_labels = PreLabels(name="cord",
                               data=X,
                               data_feats = bbox,
                               gold_labels=Y,
                               rules=rules,
                               labels_enum=ClassLabels,
                               num_classes=3)
L,S = cord_noisy_labels.get_labels()

# from spear4HighFidelity.spear.labeling import LFAnalysis
# analyse = cord_noisy_labels.analyse_lfs(plot=False)
# analyse
# from spear4HighFidelity.spear.labeling import LFAnalysis

# analyse = cord_noisy_labels.analyse_lfs(plot=True)

# result = analyse.head(16)
# display(result)

import json
import pandas as pd
import numpy as np
import os
from tqdm import tqdm_notebook as tqdm

L_w=[]
for i in range(len(token_train)):
    cord_noisy_labels_test = PreLabels(name="cord",
                               data=np.array(token_train[i]),
                               gold_labels=np.array(label_train[i]),
                               data_feats=np.array(list(boxe_train[i])),
                               rules=rules,
                               labels_enum=ClassLabels,
                               num_classes=3)
    L,S = cord_noisy_labels_test.get_labels()
    L_w.append(L)      

for i in range(len(L_w)):
    for j in range(len(L_w[i])):
        for k in range(len(L_w[i][j])):
            if L_w[i][j][k]==None:
                L_w[i][j][k]=-1

L_w_test=[]
for i in range(len(token_test)):
    cord_noisy_labels_test = PreLabels(name="cord",
                               data=np.array(token_test[i]),
                               gold_labels=np.array(label_test[i]),
                               data_feats=np.array(list(boxe_test[i])),
                               rules=rules,
                               labels_enum=ClassLabels,
                               num_classes=3)
    L,S = cord_noisy_labels_test.get_labels()
    L_w_test.append(L)      

for i in range(len(L_w_test)):
    for j in range(len(L_w_test[i])):
        for k in range(len(L_w_test[i][j])):
            if L_w_test[i][j][k]==None:
                L_w_test[i][j][k]=-1

L_w_dev=[]
for i in range(len(token_dev)):
    cord_noisy_labels_test = PreLabels(name="cord",
                               data=np.array(token_dev[i]),
                               gold_labels=np.array(label_dev[i]),
                               data_feats=np.array(list(boxe_dev[i])),
                               rules=rules,
                               labels_enum=ClassLabels,
                               num_classes=3)
    L,S = cord_noisy_labels_test.get_labels()
    L_w_dev.append(L)
          
for i in range(len(L_w_dev)):
    for j in range(len(L_w_dev[i])):
        for k in range(len(L_w_dev[i][j])):
            if L_w_dev[i][j][k]==None:
                L_w_dev[i][j][k]=-1

def org_lab(label):
    for i in range(len(label)):
        for j in range(len(label[i])):
            if (label[i][j] == 0):
                label[i][j] = 'MENU'
            elif (label[i][j] == 2):
                label[i][j] = 'QUANTITY'    
            else:
                label[i][j] = 'PRICE'
    return label            

label_train_lab=org_lab(label_train_lab)
label_train_unl=org_lab(label_train_unl)
label_test=org_lab(label_test)
label_dev=org_lab(label_dev)

token_train_lab=np.array(token_train_lab)
label_train_lab=np.array(label_train_lab)
boxe_train_lab=np.array(boxe_train_lab)
L_lab=np.array(L_w)[:lf]

token_train_unl=np.array(token_train_unl)
label_train_unl=np.array(label_train_unl)
boxe_train_unl=np.array(boxe_train_unl)
L_unl=np.array(L_w)[lf:]

token_train_dev=np.array(token_dev)
label_train_dev=np.array(label_dev)
boxe_train_dev=np.array(boxe_dev)
L_dev=np.array(L_w_dev)

token_train_test=np.array(token_test)
label_train_test=np.array(label_test)
boxe_train_test=np.array(boxe_test)
L_test=np.array(L_w_test)

import pickle
with open('Paths/cord_train_jl_1%.pkl', 'wb') as t:
    pickle.dump(np.array([token_train_lab, label_train_lab, boxe_train_lab,L_lab]), t)
with open('Paths/cord_train_u_jl_1%.pkl', 'wb') as t:
    pickle.dump(np.array([token_train_unl, label_train_unl, boxe_train_unl,L_unl]), t)    
with open('Paths/cord_val_jl_1%.pkl', 'wb') as t:
    pickle.dump(np.array([token_train_dev, label_train_dev, boxe_train_dev,L_dev]), t)
with open('Paths/cord_test_jl_1%.pkl', 'wb') as t:
    pickle.dump(np.array([token_train_test, label_train_test, boxe_train_test,L_test]), t)

import torch
import torchvision
import matplotlib.pyplot as plt
import os
import json
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import numpy as np
import train

path_json = 'Paths/cord_json.json'
T_path_pkl = 'Paths/cord_pickle_T_1%.pkl' #test data - have true labels
U_path_pkl = 'Paths/cord_pickle_U_1%.pkl' #unlabelled data - don't have true labels
# unlabelled = 'NHCages_Paths/train_U_0.1%.pkl' 
log_path_cage_1 = 'Paths/cord_log_z_1_1%.txt' #cage is an algorithm, can be found below
params_path = 'Paths/cord_params_1%.pkl' #file path to store parameters of Cage, used below
TI_path_pkl = 'Paths/cord_test_pickle_T_1%.pkl' #test data - have true labels
Z_path_pkl = 'Paths/cord_dev_pickle_U_1%.pkl' #unlabelled data - don't have true labels

n_lfs=n_lfs
n_features = 4
# 16634
n_hidden = 512
feature_model = 'layoutlm'

jl = train.JL(path_json = path_json, n_lfs = n_lfs, n_features = n_features, feature_model = feature_model)

loss_func_mask = [1,0,1,0,0,1,1] 
batch_size = 32
lr_fm = 5e-5
lr_gm = 0.01
use_accuracy_score = False

jl = train.JL(path_json = path_json, n_lfs = n_lfs, n_features = n_features, feature_model = feature_model, \
        n_hidden = n_hidden)
path_pkl = 'Paths/cord_test_jl_1%.pkl'
zpath_pkl = 'Paths/cord_val_jl_1%.pkl'
qpath_pkl = 'Paths/cord_train_jl_1%.pkl'
qpath_u_pkl = 'Paths/cord_train_u_jl_1%.pkl'

probs_fm, probs_gm = jl.fit_and_predict_proba(path_L = T_path_pkl, path_U = U_path_pkl, path_V = Z_path_pkl, \
        path_T = TI_path_pkl, train=qpath_pkl,train_u=qpath_u_pkl,dev=zpath_pkl, test=path_pkl, loss_func_mask = loss_func_mask, batch_size = batch_size, lr_fm = lr_fm, lr_gm = \
    lr_gm, use_accuracy_score = use_accuracy_score, path_log = log_path_cage_1, return_gm = True, n_epochs = \
    5, start_len = 0,stop_len = 4, is_qt = True, is_qc = True, qt = 0.9, qc = 0.85, metric_avg = 'macro')

# labels = np.argmax(probs_fm[0], 1)
print("probs_fm shape: ", probs_fm.shape)
print("probs_gm shape: ", probs_gm.shape)