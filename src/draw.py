import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from datasets.main import load_dataset
from networks.main import build_network
from datasets.main import DigitConDataset

from tqdm import tqdm
from util import plot_confusion_matrix, sim_matrix
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import numpy as np
import torchvision
import math
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import faiss
from faiss import normalize_L2
import time
import scipy
from sklearn.metrics import confusion_matrix
from sklearn.neighbors.kde import KernelDensity
from sklearn.mixture import BayesianGaussianMixture
from scipy.signal import argrelextrema
import os

from torch.autograd import Variable
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
def draw_pseudolabel_correctness_distribution(self):
    incorrect_idx = np.where(self.pl_correct_idx==0)
    correct_idx = np.where(self.pl_correct_idx==1)
    plt.hist(self.weight[incorrect_idx], bins=15,color ="red",label="incorrect",alpha=0.5, range=(0,1))
    plt.hist(self.weight[correct_idx], bins=15,color ="blue",label="correct",alpha=0.5, range=(0,1))
    plt.legend(loc='upper right')
    plt.savefig("../exp/exp_fig/pseudolabel_correctness_distribution/"+self.experiment+"_"+str(self.sub)+".svg")
    plt.close()
        
def draw_pl_confidience_feature_distribution(self):

    incorrect_idx = np.where(self.pl_correct_idx==0)
    correct_idx = np.where(self.pl_correct_idx==1)
    src_idx = list(np.random.permutation(len(self.src_dataset)))
    tar_idx = list(np.random.permutation(len(self.tar_dataset)))
    n = len(self.src_dataset) if len(self.src_dataset) < len(self.tar_dataset) else len(self.tar_dataset)
    n = n if n<1000 else 1000
    X_embedded = TSNE(n_components=2).fit_transform(np.concatenate((self.src_feature[src_idx][:n],self.tar_feature[tar_idx][:n]),axis=0))
    plt.scatter(X_embedded[:n,0],X_embedded[:n,1],c="black",label="source",marker=".",s=50)
    plt.scatter(X_embedded[n:,0],X_embedded[n:,1],c=self.weight[tar_idx][:n],label="target",marker=".",s=50)
    plt.legend(loc='upper right')
    plt.savefig("../exp/exp_fig/confidience_feature_distribution/"+self.experiment+"_"+str(self.sub)+".svg")
    plt.close()
def draw_feature_distribution(self):

    incorrect_idx = np.where(self.pl_correct_idx==0)
    correct_idx = np.where(self.pl_correct_idx==1)
    src_idx = list(np.random.permutation(len(self.src_dataset)))
    tar_idx = list(np.random.permutation(len(self.tar_dataset)))
    n = len(self.src_dataset) if len(self.src_dataset) < len(self.tar_dataset) else len(self.tar_dataset)
    n = n if n<1000 else 1000
    X_embedded = TSNE(n_components=2).fit_transform(np.concatenate((self.src_feature[src_idx][:n],self.tar_feature[tar_idx][:n]),axis=0))
    plt.scatter(X_embedded[:n,0],X_embedded[:n,1],c="red",label="source",marker=".",s=50)
    plt.scatter(X_embedded[n:,0],X_embedded[n:,1],c="blue",label="target",marker=".",s=50)
    plt.legend(loc='upper right')
    plt.savefig("../exp/exp_fig/feature_distribution/"+self.experiment+"_"+str(self.sub)+".svg")
    plt.close()

def draw_pl_confidience_feature_distribution_partialclass(self):

    for i in range(10):
        src_idx_0 = np.where(self.src_dataset.train_set.trainy==i*3)[0]
        tar_idx_0 = np.where(self.tar_dataset.train_set.trainy==i*3)[0]

        src_idx_1 = np.where(self.src_dataset.train_set.trainy==i*3+1)[0]
        tar_idx_1 = np.where(self.tar_dataset.train_set.trainy==i*3+1)[0]
        src_idx_2 = np.where(self.src_dataset.train_set.trainy==i*3+2)[0]
        tar_idx_2 = np.where(self.tar_dataset.train_set.trainy==i*3+2)[0]
        tar_idx = np.concatenate((tar_idx_0,tar_idx_1,tar_idx_2),axis=0)
        pred_0 = np.where(self.pseudo_label[tar_idx]==i*3)[0]
        pred_1 = np.where(self.pseudo_label[tar_idx]==i*3+1)[0]
        pred_2 = np.where(self.pseudo_label[tar_idx]==i*3+2)[0]
        pred_3 = np.where((self.pseudo_label[tar_idx]!=i*3+2)&(self.pseudo_label[tar_idx]!=i*3+1)&(self.pseudo_label[tar_idx]!=i*3))[0]
#             pred_other = np.where(self.pseudo_label[tar_idx]==i*3+2)[0]
#         n = len(self.src_dataset) if len(self.src_dataset) < len(self.tar_dataset) else len(self.tar_dataset)
#         n = n if n<1000 else 1000
        X_embedded = np.concatenate((self.src_feature[src_idx_0],self.src_feature[src_idx_1],self.src_feature[src_idx_2],self.tar_feature[tar_idx]),axis=0)
        X_embedded = TSNE(n_components=2,random_state=1).fit_transform(X_embedded)
        sn0 = len(src_idx_0)
        sn1 = len(src_idx_1)
        sn2 = len(src_idx_2)
        tn0 = len(tar_idx_0)
        tn1 = len(tar_idx_1)
        tn2 = len(tar_idx_2)

        plt.scatter(X_embedded[:sn0,0],X_embedded[:sn0,1],c="red",label="source class 1",marker="o",s=50)
        plt.scatter(X_embedded[sn0:sn0+sn1,0],X_embedded[sn0:sn0+sn1,1],c="blue",label="source class 2",marker="o",s=50)
        plt.scatter(X_embedded[sn0+sn1:sn0+sn1+sn2,0],X_embedded[sn0+sn1:sn0+sn1+sn2,1],c="green",label="source class 3",marker="o",s=50)

        color = ["red","blue","green","black"]
        label = ["target class 1","target class 2","target class 3","other"]
        for ii,idx in enumerate([pred_0,pred_1,pred_2,pred_3]):
            plt.scatter(X_embedded[sn0+sn1+sn2:,0][idx],X_embedded[sn0+sn1+sn2:,1][idx],c=color[ii],label=label[ii],marker="^",s=self.weight[tar_idx][idx]*50)
        plt.legend(loc='lower right')
        plt.savefig("../exp/exp_fig/confidience_feature_distribution_partialclass/"+self.experiment+"_"+str(self.sub)+"_"+str(i)+".png")
        plt.close()

        plt.scatter(X_embedded[:sn0,0],X_embedded[:sn0,1],c="red",label="source class 1",marker="o",s=50)
        plt.scatter(X_embedded[sn0:sn0+sn1,0],X_embedded[sn0:sn0+sn1,1],c="blue",label="source class 2",marker="o",s=50)
        plt.scatter(X_embedded[sn0+sn1:sn0+sn1+sn2,0],X_embedded[sn0+sn1:sn0+sn1+sn2,1],c="green",label="source class 3",marker="o",s=50)
        sn = sn0+sn1+sn2
        plt.scatter(X_embedded[sn:sn+tn0,0],X_embedded[sn:sn+tn0,1],c="red",label="target class 1",marker="^",s=50)
        plt.scatter(X_embedded[sn+tn0:sn+tn0+tn1,0],X_embedded[sn+tn0:sn+tn0+tn1,1],c="blue",label="target class 2",marker="^",s=50)
        plt.scatter(X_embedded[sn+tn0+tn1:sn+tn0+tn1+tn2,0],X_embedded[sn+tn0+tn1:sn+tn0+tn1+tn2,1],c="green",label="target class 3",marker="^",s=50)
        plt.legend(loc='lower right')
        plt.savefig("../exp/exp_fig/confidience_feature_distribution_partialclass/"+self.experiment+"_"+str(self.sub)+"_"+str(i)+"ans.png")
        plt.close()

def draw_weight_segmented_distribution(self):
    import seaborn as sns 
    sns_plot = sns.kdeplot(self.weight, shade=True, color='steelblue')
    x = sns_plot.lines[0].get_xdata() # Get the x data of the distribution
    y = sns_plot.lines[0].get_ydata() # Get the y data of the distribution
    cid = []
    for c in self.seg_point:
        cid.append(np.argmin(np.abs(x-c)))
    cid = np.int64(np.array(cid)) # The id of the peak (maximum of y data)
    print(cid)
    plt.plot(x[cid],y[cid], 'ro', ms=10)
#         plt.savefig(""+self.experiment+"_"+str(self.sub)+".png")
    plt.savefig("../exp/exp_fig/weight_segmented_distribution/"+self.experiment+"_"+str(self.sub)+".svg")
    plt.close()
def draw_portion_probability_distribution(self,portions,sort_soft):
     for i,p in enumerate(portions):
        h=np.mean(sort_soft[p],axis=0)
        plt.bar([j for j in range(h.shape[0])],h)
        plt.savefig("../exp/exp_fig/portion_probability_distribution/"+self.experiment+"_"+str(self.sub)+"_"+str(i)+".svg")
        plt.close()
def draw_portion_segmentation_feature_distribution(self,portion):
    src_idx = list(np.random.permutation(len(self.src_dataset)))
    min_l = 999999
    indx = np.empty(0)
    tar_n = []
    for i,p in enumerate(portion):
        tar_idx = list(np.random.permutation(p.shape[0]))
        if p.shape[0]>200:
            indx = np.concatenate((indx,p[tar_idx][:200]),axis=0)
            tar_n.append(200)
        else:
            indx = np.concatenate((indx,p[tar_idx][:]),axis=0)
            tar_n.append(p.shape[0])
    n = len(self.src_dataset) if len(self.src_dataset) < np.array(tar_n).sum() else np.array(tar_n).sum()
    n = n if n<1000 else 1000
    X_embedded = TSNE(n_components=2).fit_transform(np.concatenate((self.src_feature[src_idx][:n],self.tar_feature[np.int64(indx)]),axis=0))
    plt.scatter(X_embedded[:n,0],X_embedded[:n,1],c="black",label="source",marker=".",s=50)
    colors = ["green","blue","yellow","orange","red"]
    t_n = 0
    for i,p in enumerate(portion):
        l_t_n = t_n
        t_n += tar_n[i]
    plt.legend(loc='upper right')
    plt.savefig("../exp/exp_fig/portion_segmentation_feature_distribution/"+self.experiment+"_"+str(self.sub)+".svg")
    plt.close()

def draw_pl_portions_class_distribution(self):
    bottom=np.zeros(self.src_dataset.n_classes)
    for subb in range(1,self.subdomains+1):
        portion_l = int(self.weight.shape[0]*((subb-1)/self.subdomains))
        portion_r = int(self.weight.shape[0]*(subb/self.subdomains))
        index = np.argsort(-self.weight)[portion_l:portion_r].reshape(-1)
        bar_label=[]
        bar_count=[]
        for a in range(self.src_dataset.n_classes):
            bar_label.append(a)
            bar_count.append(np.int64(self.pseudo_label[index]==a).sum())
        plt.bar(bar_label,bar_count,bottom=bottom,label=subb)
        bar_count=np.array(bar_count)
        bottom=bottom+bar_count
    plt.legend(loc='upper right')
    plt.savefig("../exp/exp_fig/pl_portions_class_distribution/"+self.experiment+"_"+str(self.sub)+".svg")
    plt.close()

def draw_prediction_portions_accuracy(self):
    bar_x=[]
    bar_y=[]
    for subb in range(1,self.subdomains+1):
        portion_l = int(self.weight.shape[0]*((subb-1)/self.subdomains))
        portion_r = int(self.weight.shape[0]*(subb/self.subdomains))
        index = np.argsort(-self.weight)[portion_l:portion_r].reshape(-1)
        bar_x.append(subb)
        bar_y.append(self.prediction_correct_idx[index].mean()*100.)
    plt.bar(bar_x, bar_y)
    for index, value in enumerate(bar_y):
        plt.text(index+1, value, str(value))
    plt.savefig("../exp/exp_fig/prediction_portions_accuracy/"+self.experiment+"_"+str(self.sub)+".svg")
    plt.close()
def draw_prediction_entropy_portions_accuracy(self):
    bar_x=[]
    bar_y=[]
    for subb in range(1,self.subdomains+1):
        portion_l = int(self.ent_weight.shape[0]*((subb-1)/self.subdomains))
        portion_r = int(self.ent_weight.shape[0]*(subb/self.subdomains))
        index = np.argsort(self.ent_weight)[portion_l:portion_r].reshape(-1)
        bar_x.append(subb)
        bar_y.append(self.prediction_correct_idx[index].mean()*100.)
    plt.bar(bar_x, bar_y)
    for index, value in enumerate(bar_y):
        plt.text(index+1, value, str(value))
    plt.savefig("../exp/exp_fig/prediction_entropy_portions_accuracy/"+self.experiment+"_"+str(self.sub)+".svg")
    plt.close()
    result = np.concatenate((self.weight.reshape(-1,1),self.ent_weight.reshape(-1,1)),axis=1)
    result = np.concatenate((result,self.prediction_correct_idx.reshape(-1,1)),axis=1)
    df_result = pd.DataFrame(result,columns = ["weight","ent_weight","correct"])
    df_result.to_csv("../exp/exp_fig/prediction_accuracy_csv/"+self.experiment+"_"+str(self.sub)+".csv")

def draw_prediction_confusion(self):
    cnf_matrix = confusion_matrix(self.tar_dataset.train_set.trainy, self.prediction)
    plot_confusion_matrix(cnf_matrix,classes=[a for a in
                                              range(self.tar_dataset.n_classes)],normalize=True)
    plt.savefig("../exp/exp_fig/prediction_confusion/"+self.experiment+"_"+str(self.sub)+".svg")
    plt.close()
    
def draw_lp_vs_k(self):
    acclist=[]
    for self.k in range(5,200):
        acc=self.update_pseudo_label()
        acclist.append([self.k,acc])
    np.save(self.experiment, np.array(acclist))