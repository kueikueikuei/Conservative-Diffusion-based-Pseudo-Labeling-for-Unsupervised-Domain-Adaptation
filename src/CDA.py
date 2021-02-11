from datasets.main import load_dataset
from networks.main import build_network
from datasets.main import DigitConDataset
from loss import HLoss,mixup
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
import os
from torch.autograd import Variable
from sklearn.cluster import KMeans
import pandas as pd
from draw import *
class CDA(object):
    """A class for the Curriculum Domain Adaptation method.

    Attributes:
        self.experiment           = str   --experiment name for log file
        self.pseudo_label_method  = str   "label propagation" or "nearest center" or "NN prediction"
        self.strategy             = str   "decrementedthreshold" or "equallysegmented" or "hard2easy" or "all"
        self.schedule             = str   "log" or "linear" or "exp"
        self.subdomains           = int   --curriculum steps
        self.alpha                = float --parameter alpha for label propagation 
        self.k                    = int   --parameter k for label propagation
        self.init                 = float --init threshold
        self.src_w                = float --parameter for source loss
        self.tar_w                = float --parameter for target loss 
        self.mft_w                = float --parameter for mixup feature loss
        self.epoch                = int   --parameter for training epoch in each easy target adaptation
        build_default_parameter:  build network, batchsize, and optimizer according dataset
    """

    def __init__(self, src = "amazon", tar = "webcam", pseudo_label_method = "lp", strategy = "easy2hard",schedule = None, subdomains = 5, experiment = "", epoch = 10, alpha = 0.5, src_w = 1, tar_w = 1, init = 0.1, mft_w = 0.1,k=50,record=True,device="cuda:0"):
        
        ## parameter
        self.experiment = experiment
        self.pseudo_label_method = pseudo_label_method
        self.strategy = strategy
        self.subdomains = subdomains
        self.k = k
        self.alpha = alpha
        self.src = src
        self.tar = tar
        self.src_w = src_w
        self.tar_w = tar_w
        self.mft_w = mft_w
        self.init = init
        self.device = device
        self.record = record
        self.epoch = epoch
        self.schedule = schedule
        ## record pseudo label
        self.weight = None
        self.pseudo_label = None
        self.src_feature = None
        self.tar_feature = None
        self.seg_point = None
        self.globalstep = 0
        
        ## data
        self.src_dataloader = None
        self.tar_dataloader = None
        self.src_dataloader_shuffle = None
        self.tar_dataloader_shuffle = None
        self.src_dataset = load_dataset(src)
        self.tar_dataset = load_dataset(tar)
        self.easy_tar_dataset = load_dataset(tar)
        ## model
        self.build_default_parameter()
        self.load_dataloader()
        
        
        
        ## log files
        self.writer = SummaryWriter("../log/"+experiment)
        fp = open('experiments.txt', "r")
        for i in iter(fp):
            if not os.path.exists('../exp/exp_fig/'+i+'/'):
                os.makedirs('../exp/exp_fig/'+i+'/')
        fp.close()
        ## loss
        self.ce = torch.nn.NLLLoss()
        self.ce_noreduce = torch.nn.NLLLoss(reduce=False)
        self.ent = HLoss()
        
    def train(self):
        easy_tar_dataloader = None
        if self.pseudo_label_method == "entropy":
            self.pretrain()
            self.sub_da_testing()
            return
        
        for self.sub in range(self.subdomains):
            self.update_pseudo_label()
            draw_feature_distribution(self)
#             self.draw_lp_vs_k()
            if self.record:
                draw_pseudolabel_correctness_distribution(self)
                draw_pl_confidience_feature_distribution_partialclass(self)
                draw_pl_confidience_feature_distribution(self)
                draw_pl_portions_class_distribution(self)
            easy_tar_dataloader = self.easy_target_selection()
            if easy_tar_dataloader == None:
                return
            if self.record:
                draw_weight_segmented_distribution(self)
            
            for e in range(self.epoch):
                self.easy_target_adaptation(easy_tar_dataloader,self.src_w,self.tar_w,self.mft_w)
                self.adapted_network_testing()
            if self.record:    
                draw_prediction_entropy_portions_accuracy(self)
                draw_prediction_portions_accuracy(self)
                draw_prediction_confusion(self)

    def build_default_parameter(self):
        if self.src_dataset.domain=="office31":
            self.batchsize = 25
            lr=0.001
            momentum=0.9
            l2_decay=5e-4
            self.net = build_network("office_ResNet50").to(self.device)
            self.optimizer = torch.optim.SGD([
            {'params': self.net.backbone.parameters(), 'lr': lr},
            {'params': self.net.classifier1.parameters(), 'lr': lr*10}
            ], momentum=momentum, weight_decay=l2_decay)
        elif self.src_dataset.domain=="visda":
            self.batchsize = 50
            lr=0.001
            momentum=0.9
            l2_decay=5e-4
            self.net = build_network("visda_ResNet50")
            self.optimizer = torch.optim.SGD([
            {'params': self.net.backbone.parameters(), 'lr': lr},
            {'params': self.net.classifier1.parameters(), 'lr': lr*10}
            ], momentum=momentum, weight_decay=l2_decay)
            
    def load_dataloader(self):
        self.src_dataloader_shuffle = self.src_dataset.loaders(batch_size=self.batchsize, num_workers=8)
        self.tar_dataloader_shuffle = self.tar_dataset.loaders(batch_size=self.batchsize, num_workers=8)
        self.src_dataloader = self.src_dataset.loaders(batch_size=self.batchsize, shuffle=False, num_workers=8)
        self.tar_dataloader = self.tar_dataset.loaders(batch_size=self.batchsize, shuffle=False, num_workers=8)

    
    def update_pseudo_label(self):
        if self.pseudo_label_method == "label propagation":
            self.weight, self.pseudo_label ,self.src_feature, self.tar_feature = self._plabels_label_propagation(alpha = self.alpha, k = self.k, max_iter = 20,layer = 0)
        elif self.pseudo_label_method == "nearest center":
            self.weight, self.pseudo_label, self.src_feature, self.tar_feature = self._plabels_nearest_center()
        elif self.pseudo_label_method == "NN prediction":
            self.weight, self.pseudo_label, self.src_feature, self.tar_feature = self._plabels_NN_prediction()
        self.pl_correct_idx = np.int64(self.tar_dataset.train_set.trainy == self.pseudo_label)
        self.writer.add_scalar('pseudo label acc', 100.*self.pl_correct_idx.mean(), self.sub)
        print("pseudo label accuracy",100.*self.pl_correct_idx.mean())  
        return 100.*self.pl_correct_idx.mean()

    def easy_target_selection(self):
        if self.sub!=0: ### record label propagation imformation before after adaptation
            hardcorrect_idx = np.int64(self.tar_dataset.train_set.trainy[self.hard_index] == self.pseudo_label[self.hard_index])
            wrong = np.where((hardcorrect_idx-self.hardcorrect_idx)<0)[0].shape[0]
            correct = np.where((hardcorrect_idx-self.hardcorrect_idx)>0)[0].shape[0]
            easycertainty = self.weight[self.easy_index]
            self.writer.add_scalar('wrong hard pseudo label acc after', wrong, self.sub)
            self.writer.add_scalar('correct hard pseudo label acc after', correct, self.sub)
            self.writer.add_scalar('easy certainty after', 100.*easycertainty.mean(), self.sub)

        portions = []
        
        if self.strategy=="equallysegmented":
            for i in range(self.subdomains):
                portions.append(np.argsort(-self.weight)[int(self.weight.shape[0]*((i)/self.subdomains)):int(self.weight.shape[0]*((i+1)/self.subdomains))].reshape(-1))
            easy_index = np.argsort(-self.weight)[:int(self.weight.shape[0]*((self.sub+1)/self.subdomains))].reshape(-1)
            c = []
            for i in range(self.subdomains-1):
                c.append(-np.sort(-self.weight)[int(self.weight.shape[0]*((i+1)/self.subdomains))].reshape(-1))
            self.easy_index = easy_index
            self.hard_index = np.argsort(-self.weight)[int(self.weight.shape[0]*((self.sub+1)/self.subdomains)):].reshape(-1)
            self.seg_point = np.array(c)
        elif self.strategy == "decrementedthreshold":
            if "log" == self.schedule:
                t = 1-((1-math.exp((-self.sub/self.subdomains)*5))*(self.init)+(1-self.init))
            elif "linear" == self.schedule:
                t = 1-(self.sub/self.subdomains*(self.init)+(1-self.init))
            elif "exp" == self.schedule:
                t = 1-(math.exp((self.sub/self.subdomains-1)*5)*(self.init)+(1-self.init))
            print(t)
            easy_index = np.where(self.weight>t)
            self.easy_index = easy_index
            self.hard_index = np.where(self.weight<t)
            self.seg_point = np.array([t])
        elif self.strategy=="hard2easy":
            for i in range(self.subdomains):
                portions.append(np.argsort(self.weight)[int(self.weight.shape[0]*((i)/self.subdomains)):int(self.weight.shape[0]*((i+1)/self.subdomains))].reshape(-1))
            easy_index = np.argsort(self.weight)[:int(self.weight.shape[0]*((self.sub+1)/self.subdomains))].reshape(-1)
            c = []
            for i in range(self.subdomains-1):
                c.append(np.sort(self.weight)[int(self.weight.shape[0]*((i+1)/self.subdomains))].reshape(-1))
            self.seg_point = np.array(c)
        elif self.strategy == "all":
            easy_index = np.argsort(-self.weight)[:].reshape(-1)
            self.easy_index = easy_index
            self.hard_index = easy_index
            self.seg_point = np.array([0])
        self.easy_tar_dataset.train_set.conf = self.weight[easy_index]
        self.easy_tar_dataset.train_set.ps = self.pseudo_label[easy_index]
        self.easy_tar_dataset.train_set.img = self.tar_dataset.train_set.img[easy_index]
        self.easy_tar_dataset.train_set.trainy = self.tar_dataset.train_set.trainy[easy_index]
        try:
            easy_tar_dataloader = self.easy_tar_dataset.loaders(batch_size=self.batchsize, num_workers=0)
        except:
            return None
        easycorrect_idx = np.int64(self.tar_dataset.train_set.trainy[easy_index] == self.pseudo_label[easy_index])
        self.writer.add_scalar('easy pseudo label acc', 100.*easycorrect_idx.mean(), self.sub)
        print("easy pseudo label accuracy",100.*easycorrect_idx.mean(),"numbers ",easycorrect_idx.shape)
        self.hardcorrect_idx = np.int64(self.tar_dataset.train_set.trainy[self.hard_index] == self.pseudo_label[self.hard_index])
        easycertainty = self.weight[self.easy_index]
        self.writer.add_scalar('hard pseudo label acc before', 100.*self.hardcorrect_idx.sum(), self.sub)
        self.writer.add_scalar('easy certainty before', 100.*easycertainty.mean(), self.sub)
        
        return easy_tar_dataloader
    def pretrain(self):
#         self.net.load_state_dict(torch.load("../model/pretrain_"+self.src))
        iteration=int(len(self.tar_dataset)/self.batchsize)
        epoch=10
        lr=0.001
        momentum=0.9
        l2_decay=5e-4
        optimizer = torch.optim.SGD([
                {'params': self.net.classifier1.parameters(), 'lr': lr*10}
                ], momentum=momentum, weight_decay=l2_decay)
        self.net.eval()
        for name, p in self.net.named_parameters():
            if "backbone" in name:
                p.requires_grad = False
#             param.requires_grad = False
        for e in range(epoch):
            print(e)
            
            for src_data, src_label, _, _, _ in self.src_dataloader_shuffle:
                src_data, src_label = src_data.to(self.device), src_label.to(self.device)
                feature = self.net.backbone(src_data)
                src_logit = self.net.classifier1(feature)
                src_out_NLL = F.log_softmax(src_logit, 1)
                source_distance_loss=self.ce(src_out_NLL,src_label)
                source_distance_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        for name, p in self.net.named_parameters():
            if "backbone" in name:
                p.requires_grad = True

    def easy_target_adaptation(self,easy_tar_dataloader,src_w,tar_w,mft_w):
        iteration=int(len(self.tar_dataset)/self.batchsize)
        self.net.train()
        
        for i in range(1, iteration+1):
            try:
                src_data, src_label, _, _, _ = src_iter.next()
            except Exception as err:
                src_iter=iter(self.src_dataloader_shuffle)
                src_data, src_label, _, _, _  = src_iter.next()

            try:
                tar_data_easy, tar_label_easy, weight, _, plabel = tar_iter_easy.next()
            except Exception as err:
                tar_iter_easy = iter(easy_tar_dataloader)
                tar_data_easy, tar_label_easy, weight, _, plabel = tar_iter_easy.next()

            try:
                tar_data, tar_label, _, _, _ = tar_iter.next()
            except Exception as err:
                tar_iter=iter(self.tar_dataloader_shuffle)
                tar_data, tar_label, _, _, _ = tar_iter.next()

            src_data, src_label = src_data.to(self.device), src_label.to(self.device)
            tar_data,tar_label = tar_data.to(self.device),tar_label.to(self.device)
            tar_data_easy,tar_label_easy = tar_data_easy.to(self.device),tar_label_easy.to(self.device) 
            weight = weight.to(self.device)
            plabel = plabel.to(self.device)
            
            ### source task specific loss
            feature1 = self.net.backbone(src_data)
            src_logit = self.net.classifier1(feature1)
            srcprob = F.softmax(src_logit,1)
            src_out_NLL = F.log_softmax(src_logit, 1)
            source_distance_loss=self.ce(src_out_NLL,src_label)
            
            ### weighted pseudo-labeling loss
            tar_feature_easy1 = self.net.backbone(tar_data_easy)
            tar_logit_easy = self.net.classifier1(tar_feature_easy1)
            tar_out_NLL = F.log_softmax(tar_logit_easy, 1)            
            target_distance_loss = torch.sum(weight*self.ce_noreduce(tar_out_NLL,plabel))/(torch.sum(weight)+0.00001)

            ### mixup-feature training loss
            tar_feature1 = self.net.backbone(tar_data)
            sn,sd = feature1.shape
            tn,td = tar_feature1.shape
            mix = mixup( torch.cat((src_data[:int(sn/2)],tar_data[:int(tn/2)]),0), torch.cat((feature1[:int(sn/2)],tar_feature1[:int(tn/2)]),0).detach(),self.net.backbone,0.1)
            
            ### total loss
            loss_total = 0.5 * source_distance_loss + target_distance_loss + mft_w*mix
            
            loss_total.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            if self.globalstep % 20 == 0:
                self.writer.add_scalar('source_distance_loss', source_distance_loss.item(), self.globalstep)
                self.writer.add_scalar('target_distance_loss', target_distance_loss.item(),  self.globalstep)
                self.writer.add_scalar('mix', mix.item(),  self.globalstep)
                self.writer.add_scalar('total_loss', loss_total.item(),  self.globalstep)
            self.globalstep+=1
            
    def adapted_network_testing(self):
        self.net.eval()
        src_dataset_len = len(self.src_dataset)
        src_correct=0
        with torch.no_grad():
            for i,(src_data, src_label,_,_,_) in enumerate(self.src_dataloader):
                src_data, src_label = src_data.to(self.device), src_label.to(self.device)
                feature = self.net.backbone(src_data)
                logit = self.net.classifier1(feature)
                prob = F.softmax(logit, 1)
                pred = prob.data.max(1)[1] # get the index of the max log-probability
                src_correct += pred.eq(src_label.data.view_as(pred)).cpu().sum()
        print("src acc",100. * src_correct/src_dataset_len)

        test_loss = 0
        tar_correct = 0
        tar_dataset_len = len(self.tar_dataset)
        correct_idx,preds,ent_weight = [],[],[]
        with torch.no_grad():
            for tar_data, tar_label,_,_,_ in self.tar_dataloader:
                tar_data, tar_label = tar_data.to(self.device), tar_label.to(self.device)
                feature = self.net.backbone(tar_data)
                logit = self.net.classifier1(feature)
                test_loss += F.nll_loss(F.log_softmax(logit, dim = 1), tar_label, reduction='sum').item()
                prob = F.softmax(logit, 1)
                pred = prob.data.max(1)[1] # get the index of the max log-probability
                ent_weight.append(self.ent(logit).data.cpu())
                correct_idx.append(pred.eq(tar_label.data.view_as(pred)).data.cpu())
                tar_correct += pred.eq(tar_label.data.view_as(pred)).cpu().sum()
                preds.append(pred.data.cpu())
        self.ent_weight = np.asarray(torch.cat(ent_weight).numpy())
        self.prediction_correct_idx = np.asarray(torch.cat(correct_idx).numpy())
        self.prediction = np.asarray(torch.cat(preds).numpy())
        print(self.tar+" Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)"
              .format(test_loss/tar_dataset_len, tar_correct, tar_dataset_len,100. * tar_correct / tar_dataset_len))  
        self.writer.add_scalar('acc_'+str(self.src)+"_to_"+str(self.tar), 100. * tar_correct / tar_dataset_len, self.globalstep)
    
    def _plabels_NN_prediction(self):
        self.net.eval()
        embeddings_src,embeddings_tar,preds,ent_weight=[],[],[],[]
        with torch.no_grad():
            for src_input, _, _, _, _ in self.src_dataloader:
                src_input = src_input.to(self.device)
                feature = self.net.backbone(src_input)
                embeddings_src.append(feature.data.cpu())
        with torch.no_grad():
            for tar_data, tar_label,_,_,_ in self.tar_dataloader:
                tar_data, tar_label = tar_data.to(self.device), tar_label.to(self.device)
                feature = self.net.backbone(tar_data)
                embeddings_tar.append(feature.data.cpu())
                logit = self.net.classifier1(feature)
                prob = F.softmax(logit, 1)
                pred = prob.data.max(1)[1] # get the index of the max log-probability
                ent_weight.append((1-self.ent(logit)).data.cpu())
                preds.append(pred.data.cpu())
        pss = np.asarray(torch.cat(preds).numpy())
        coss = np.asarray(torch.cat(ent_weight).numpy())

        coss = (coss-coss.min()) /(coss.max()-coss.min())

        src_X = np.asarray(torch.cat(embeddings_src).numpy())
        tar_X = np.asarray(torch.cat(embeddings_tar).numpy())

        return coss,pss,src_X,tar_X
    
    def _plabels_nearest_center(self):
        c = self._init_center(self.src_dataloader,self.src_dataset.n_classes)
        self.net.eval()
        embeddings_src,embeddings_tar,pss,coss=[],[],[],[]
        for src_input, _, _, _, _ in self.src_dataloader:
            src_input = src_input.to(self.device)
            feature = self.net.backbone(src_input)
            embeddings_src.append(feature.data.cpu())
        for tar_input, _, _, _, _ in self.tar_dataloader:
            tar_input = tar_input.to(self.device)
            feature = self.net.backbone(tar_input)
            embeddings_tar.append(feature.data.cpu())
            cosine, cos_psuedo_label = torch.max(sim_matrix(feature,c),axis=1)
            coss.append(cosine.data.cpu())
            pss.append(cos_psuedo_label.data.cpu())

        pss = np.asarray(torch.cat(pss).numpy())
        coss = np.asarray(torch.cat(coss).numpy())
        coss = (coss-coss.min()) /(coss.max()-coss.min())

        src_X = np.asarray(torch.cat(embeddings_src).numpy())
        tar_X = np.asarray(torch.cat(embeddings_tar).numpy())

        return coss,pss,src_X,tar_X
    
    def _init_center(self,src_dataloader,classnum):
        self.net.eval()
        center = torch.zeros((classnum,self.net.rep_dim)).to(self.device)
        size = torch.zeros((classnum)).to(self.device)
        
        for step,(src_input, src_label,_,_,_) in enumerate(src_dataloader):
            src_input = src_input.to(self.device)
            src_label = src_label.to(self.device).reshape(-1)
            feature = self.net.backbone(src_input)
            size = size.scatter_add_(0, src_label, torch.ones_like(size).data)
            center = center.scatter_add_(0, src_label.unsqueeze(1).expand(feature.size()), feature.data)
        return center / size.reshape(-1, 1)
    def _plabels_label_propagation(self,alpha = 0.50, k = 50, max_iter = 100,layer=0):
        
        embeddings_src,embeddings_tar,labels_src=[],[],[]
        embeddings_tar_adv = []
        self.net.eval()
        with torch.no_grad():
            if layer==0:
                model = self.net.backbone
            else:
                model = torch.nn.Sequential(*list(self.net.backbone.children())[:-layer])
            for src_input, src_label,_,_,_ in tqdm(self.src_dataloader):
                src_input = src_input.to(self.device,non_blocking=True)
                src_label = src_label.to(self.device,non_blocking=True)
                feature = model(src_input).reshape(src_input.shape[0],-1)
                embeddings_src.append(feature.data.cpu())
                labels_src.append(src_label.data.cpu())
            for tar_input, _,_,_,_ in tqdm(self.tar_dataloader):
                tar_input = tar_input.to(self.device,non_blocking=True)
                feature = model(tar_input).reshape(tar_input.shape[0],-1)
                embeddings_tar.append(feature.data.cpu())

            src_X = np.asarray(torch.cat(embeddings_src).numpy())
            labels = np.asarray(torch.cat(labels_src).numpy())
            tar_X = np.asarray(torch.cat(embeddings_tar).numpy())
        
        embeddings_src_len = src_X.shape[0]
        embeddings_tar_len = tar_X.shape[0]
        X = np.concatenate((src_X,tar_X),axis=0)
        print('Updating pseudo-labels...')
        labeled_idx = np.asarray([a for a in range(embeddings_src_len)])
        unlabeled_idx = np.asarray([embeddings_src_len+a for a in range(embeddings_tar_len)])

        with torch.cuda.device(self.device):
            torch.cuda.empty_cache()
        # kNN search for the graph
        d = X.shape[1]
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = int(self.device[-1])
        index = faiss.GpuIndexFlatIP(res,d,flat_config)   # build the index

        normalize_L2(X)
        index.add(X) 
        N = X.shape[0]
        Nidx = index.ntotal

        c = time.time()
        D, I = index.search(X, k + 1)
        elapsed = time.time() - c
        print('kNN Search done in %d seconds' % elapsed)

        # Create the graph
        D = D[:,1:] ** 3
        I = I[:,1:]
        row_idx = np.arange(N)
        row_idx_rep = np.tile(row_idx,(k,1)).T
        W = scipy.sparse.csr_matrix((D.flatten('F'), (row_idx_rep.flatten('F'), I.flatten('F'))), shape=(N, N))
        W = W + W.T

        # Normalize the graph
        W = W - scipy.sparse.diags(W.diagonal())
        S = W.sum(axis = 1)
        S[S==0] = 1
        D = np.array(1./ np.sqrt(S))
        D = scipy.sparse.diags(D.reshape(-1))
        Wn = D * W * D

        # Initiliaze the y vector for each class (eq 5 from the paper, normalized with the class size) and apply label propagation
        Z = np.zeros((N,self.src_dataset.n_classes))
        A = scipy.sparse.eye(Wn.shape[0]) - alpha * Wn

        for i in range(self.src_dataset.n_classes):
            cur_idx = labeled_idx[np.where(labels ==i)]
            y = np.zeros((N,))
            y[cur_idx] = 1.0 / cur_idx.shape[0]
            f, _ = scipy.sparse.linalg.cg(A, y, tol=1e-6, maxiter=max_iter)
            Z[:,i] = f
        
        # Handle numberical errors
        Z[Z < 0] = 0 
        # Compute the weight for each instance based on the entropy (eq 11 from the paper)
        probs_l1 = F.normalize(torch.tensor(Z),1).numpy()
        probs_l1[probs_l1 <0] = 0
        sortprob = -np.sort(-probs_l1,axis=1)
        probsum = np.sum(probs_l1,axis=1)
        entropy = scipy.stats.entropy(probs_l1.T)
        entropy[np.where(probsum==0)]=np.log(self.src_dataset.n_classes)
        weights = 1 - entropy / np.log(self.src_dataset.n_classes)
        weights = weights / np.max(weights)
        p_labels = np.argmax(probs_l1,1)
        conf = weights[unlabeled_idx]
        conf = (conf-np.min(conf)) /(np.max(conf)-np.min(conf))
        pss = p_labels[unlabeled_idx]

        return conf,pss,src_X,tar_X
