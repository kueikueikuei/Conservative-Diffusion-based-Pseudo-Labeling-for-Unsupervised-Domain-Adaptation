from CDA import CDA
'''
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
'''

pseudo_label_method = "label propagation"
subdomains = 45
strategy = "decrementedthreshold"
schedule = "exp"
alpha = 0.5
t_list = ["amazon","webcam","amazon","dslr","webcam","dslr"]
s_list = ["webcam","amazon","dslr","webcam","dslr","amazon"]
k=50
mft_w = 1
init = 0.7
src_w = 0.5
tar_w = 1
for s,t in zip(s_list,t_list):
    if (s=="webcam" and t=="dslr") or (s=="dslr" and t=="webcam"):
        k=6
    if (s=="webcam" and t=="amazon") or (s=="dslr" and t=="amazon"):
        mft_w=0.1
        init=0.8
    experiment = s+"_"+t+"_"+pseudo_label_method+"_"+strategy+"_"+schedule+"_init"+str(init)
    experiment += "_steps"+str(subdomains)+"_alpha"+str(alpha)+"_k"+str(k)+"_src_w"+str(src_w)+"_tar_w"+str(tar_w)+"_mft_w"+str(mft_w)
    cda = CDA(src = s, tar = t, pseudo_label_method = pseudo_label_method, strategy = strategy, schedule = schedule,\
              subdomains = subdomains, experiment = experiment, epoch = 10, alpha = alpha, src_w = src_w, tar_w = tar_w, init = init, \
              mft_w = mft_w, k = k, record = True, device = "cuda:1")
    cda.train()
    
    
    
    
    
t = "visda_tar"
s = "visda_src"
k=500
mft_w = 0.1
src_w = 0.5
tar_w = 1
init = 0.9
experiment = s+"_"+t+"_"+pseudo_label_method+"_"+strategy+"_"+schedule+"_init"+str(init)
experiment += "_steps"+str(subdomains)+"_alpha"+str(alpha)+"_k"+str(k)+"_src_w"+str(src_w)+"_tar_w"+str(tar_w)+"_mft_w"+str(mft_w)
cda = CDA(src = s, tar = t, pseudo_label_method = pseudo_label_method, strategy = strategy, schedule = schedule,\
          subdomains = subdomains, experiment = experiment, epoch = 10, alpha = alpha, src_w = src_w, tar_w = tar_w, init = init, \
          mft_w = mft_w, k = k, record = True, device = "cuda:1")
cda.train()