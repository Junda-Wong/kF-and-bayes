import numpy as np
import time
from pathlib import Path
import warnings
##试验fd对偶解和带参数的fd惩罚
#第一批数据为0时，空模型更新
class edRVFL_ridge_classification:
    def __init__(self, L=10, N=100, C=4, scale_w=1.0, scale_b=0.2, init_por=0.5, batch_por=0.1,\
                 data_norm=0, add_val=False,  use_uniform=0, use_norm=0, active_func=1, fd_first_use_ridge=False, fd_extra_update=0,\
                 voting=0, seed=1, dual_solution=False, update_mode=False,fdk=0.6):
        super().__init__()
        self.L = L#层数
        self.N = N#节点数
        self.C = 2**C#岭回归闭式解参数1/C
        self.scale_w = scale_w#随机权值放缩尺度
        self.scale_b = scale_b
        
        self.class_num=1#类别数
        self.init_por = init_por#第一批比例
        self.batch_por = batch_por#增量批比例
        self.fdk=fdk
        
        self.data_norm = data_norm
        self.add_val = add_val#是否加入验证集训练
        #self.use01 = use01#是否仿射变换
        self.use_uniform = use_uniform#是否使用均匀分布
        self.use_norm = use_norm#是否特征归一化及归一化方式:0关闭/1单独/2延续第一次
        self.active_func = active_func#激活函数类型
        self.fd_first_use_ridge = fd_first_use_ridge#fd的第一项采用岭回归
        #fd多更新一次
        self.fd_extra_update = fd_extra_update#0关闭，fd少更新一次，均采用最后更新参数；1关闭，fd少更新一次，ridge采用倒数第二次参数；
        #2开启，不论数据集是否整除，fd添加的batch同上一批一样，均采用最后更新参数#3开启，fd添加的batch随机（但是个数要手动指定），均采用最后更新参数
        self.extra_times=1
        self.voting = voting#0每个分类器最大值投票#1累加scores取最大#2累加scores分别投票
        
        self.w = []
        self.b = []
        self.beta = []
        self.beta_fd = []
        
        self.mu = []
        self.sigma = []

        np.random.seed(seed)
        
        self.dual_solution = dual_solution#是否允许用fd对偶解。如果用，给出第一批占比。(注意对偶解是样本数<层总特征数时开启)
        self.update_mode = update_mode
        #信息输出
        self.error_times = False
        self.print_info = [' ']
        self.solution_info = [' ']
        
        #计算误差
        #self.regret_control=regret_control#0:分类器分别预测，取平均然后计算误差；1：分别计算误差，误差取平均
        #self.IR_ridge_tr=[]#取成平均误差
        #self.IR_forward_tr=[]
        #self.IR_ridge_tr_all=[]
        #self.IR_forward_tr_all=[]
        self.theta_array = []
        self.theta_fd_array = []
        #准确率
        self.IR_ridge_te=[]#取成平均误差
        self.IR_forward_te=[]
        
        self.IR_ridge_sp_te=[]#分层但不叠加
        self.IR_forward_sp_te=[]
        
        self.IR_ridge_spp_te=[]#分层且叠加
        self.IR_forward_spp_te=[]
        #误差
        self.IR_ridge_err=[]#取成平均误差
        self.IR_forward_err=[]
    def train(self, x_tr, y_tr, x_val, y_val, x_te, y_te):
        start_time = time.perf_counter()

        ##是否加入验证集训练
        if self.add_val:
            X=np.concatenate((x_tr, x_val), 0)
            Y=np.concatenate((y_tr, y_val), 0)
        else:
            X=np.copy(x_tr)
            Y=np.copy(y_tr)
        del x_tr, y_tr#, x_val, y_val#仅保存X,Y,xte,yte
        
        #normalization(可选优化):训练集传播到测试集。原程序训练集测试集统一zscore
        if self.data_norm == 1:#传播法
            mean_x=np.mean(X,0)
            std_x=np.std(X,axis=0,ddof=1)###
            std_x[np.where(std_x==0)]=1e-4
            X=(X-mean_x)/np.tile(std_x,(np.shape(X)[0],1))
            x_val=(x_val-mean_x)/np.tile(std_x,(np.shape(x_val)[0],1))
            x_te=(x_te-mean_x)/np.tile(std_x,(np.shape(x_te)[0],1))
        if self.data_norm == 2:#一起zscore法
            x_num=X.shape[0]
            XX=np.concatenate((X,x_te),0)
            mean_xx=np.mean(XX,0)
            std_xx=np.std(XX,axis=0,ddof=1)###
            std_xx[np.where(std_xx==0)]=1e-4
            XX=(XX-mean_xx)/np.tile(std_xx,(np.shape(XX)[0],1))
            X=XX[:x_num]
            x_te=XX[x_num:]
            del XX
            
        ##线性变换(可选优化):训练集测试集统一变换。or 训练集传播到测试集 
        '''Xnorm=0
        if self.use01:
            Xnorm=np.concatenate((X,x_te),0)
            Xnorm=(Xnorm+np.min(Xnorm,0))/np.max(Xnorm,0)#注意这个线性变换，不是归一化。可以加上权值后统一smac搜索
            X=np.copy(Xnorm[0:X.shape[0]])
            x_te=np.copy(Xnorm[X.shape[0]:])
        del Xnorm'''

        n_sample = X.shape[0]
        n_feature = X.shape[1]
        n_class = Y.shape[1]
        if self.update_mode == True:#采用初始化更新
            if self.dual_solution==False:#不用fd对偶解时，实际上要求n_sample>总n_feature.但难以实现，所以只能暴力求近似解。使用smac优化
                init_num = int(np.floor(self.init_por * n_sample))#初始数量
                init_num = max(init_num, n_feature+1)#不允许过小
                batch_num = int(np.floor(self.batch_por * n_sample))
                batch_num = max(batch_num, 5)
            else:
                init_num = int(np.floor(self.batch_por * n_sample))
                init_num = max(init_num, 8)
                batch_num = int(np.floor(self.batch_por * n_sample))
                batch_num = max(batch_num, 8)
        else:
            init_num = int(np.floor(self.batch_por * n_sample))
            init_num = max(init_num, 1)#n_feature+1
            batch_num = int(np.floor(self.batch_por * n_sample))
            batch_num = max(batch_num, 1)#n_feature+1
        last_num = (n_sample-init_num)%batch_num#最后一批数量
        it = int((n_sample-last_num-init_num)/batch_num)#中间batch迭代次数
        
        fix_last_num=False#最后一批补全(补全一个批次,  可选). init_num=batch_num:允许使用对偶解；
        if last_num<(batch_num) and last_num>0:
            fix_last_num=True
            last_num=int(batch_num)
        
        if last_num==0:#fd更新使用的额外批次
            it=it+1
            extra_num=batch_num
        else:
            it=it+2
            extra_num=last_num
            
        #权值初始化..kaiming normal？
        if self.use_uniform==0:
            weights1=self.scale_w * (2 * np.random.random([n_feature, self.N]) - 1)#初始化L层的权值，对t个时刻保持不变
            bias1=self.scale_b * np.random.random([1, self.N])
            weights2_L=[self.scale_w * (2 * np.random.random([n_feature+self.N, self.N]) - 1) for i in range(self.L-1)]
            bias2_L=[self.scale_b * np.random.random([1, self.N]) for i in range(self.L-1)]
        elif self.use_uniform==1:
            weights1=self.scale_w * (1 * np.random.normal(0,1,[n_feature, self.N]) - 0)
            bias1=self.scale_b * np.random.normal(0,1,[1, self.N])
            weights2_L=[self.scale_w * (1 * np.random.normal(0,1,[n_feature+self.N, self.N]) - 0) for i in range(self.L-1)]
            bias2_L=[self.scale_b * np.random.normal(0,1,[1, self.N]) for i in range(self.L-1)]
        elif self.use_uniform==2:
            weights1=self.scale_w * (self.xavier_init(n_feature, self.N))
            bias1=self.scale_b * (self.xavier_init(1, self.N))
            weights2_L=[self.scale_w * (self.xavier_init(n_feature+self.N, self.N) ) for i in range(self.L-1)]
            bias2_L=[self.scale_b * self.xavier_init(1, self.N) for i in range(self.L-1)]
        elif self.use_uniform==3:
            weights1=self.scale_w * (self.kaiming_init(n_feature, self.N))
            bias1=self.scale_b * (self.kaiming_init(1, self.N))
            weights2_L=[self.scale_w * (self.kaiming_init(n_feature+self.N, self.N) ) for i in range(self.L-1)]
            bias2_L=[self.scale_b * self.kaiming_init(1, self.N) for i in range(self.L-1)]
        #普通的ridge
        theta=np.zeros((self.L,n_feature+self.N+1, n_class))#临时存储L层线性回归参数
        eta=np.zeros((self.L,n_feature+self.N+1,n_feature+self.N+1))
        Theta=[]#存储t个时刻的theta
        Eta=[]#存储t个时刻的eta
        mu = np.zeros((it,self.L,self.N), dtype=np.float64)
        sigma = np.zeros((it,self.L,self.N), dtype=np.float64)
        
        #forward  是否为其再添加一个批次的数据？
        theta_fd=np.zeros((self.L,n_feature+self.N+1, n_class))#临时存储L层线性回归参数
        eta_fd=np.zeros((self.L,n_feature+self.N+1,n_feature+self.N+1))
        eta_k=np.zeros((self.L,n_feature+self.N+1,n_feature+self.N+1))
        Theta_fd=[]#存储t个时刻的theta
        Eta_fd=[]#存储t个时刻的eta
        #mu_fd = np.zeros((it,self.L,self.N), dtype=np.float64)
        #sigma_fd = np.zeros((it,self.L,self.N), dtype=np.float64)
        
        
        ##输出特征
        F=[]#it@L@batch*(feature_num+N+1)个值
        
        YY=[]#it@batch*(class_num)
        for t in range(it):#原始数据批次：0~it(共it+1次)；额外补充一次随机抽取供forward使用,为了forward也可以更新相同的次数
            
            if t==0:
                x_train=X[0:init_num]
                y_train=Y[0:init_num]
            elif t==it-1:
                x_train=X[(init_num+(t-1)*batch_num):]
                y_train=Y[(init_num+(t-1)*batch_num):]
                if fix_last_num:
                    fix_order=np.random.randint(0,n_sample,(last_num-x_train.shape[0]))
                    x_train=np.concatenate((x_train, X[fix_order]), 0)
                    y_train=np.concatenate((y_train, Y[fix_order]), 0)
            else:
                x_train=X[(init_num+(t-1)*batch_num):(init_num+(t)*batch_num)]
                y_train=Y[(init_num+(t-1)*batch_num):(init_num+(t)*batch_num)]
        
            a_input = np.copy(x_train)
            f=[]#L@batch*(feature_num+N+1)
            #获取当前批次特征
            for i in range(self.L):#每一批数据进来都要更新网络
                if i==0:
                    a1 = np.dot(a_input, weights1) + bias1
                else:
                    a1 = np.dot(a_input, weights2_L[i-1]) + bias2_L[i-1]
            
                #normalize：每一批数据进来，提取特征，单独归一化
                if self.use_norm==0:
                    pass
                if self.use_norm==1:#每批单独归一化
                    
                    mu1 = a1.mean(0)
                    sigma1 = a1.std(0)
                    sigma1 = np.maximum(sigma1, 0.0001)  # for numerical stability
                    #mu[t,i,:]=np.copy(mu1)
                    #sigma[t,i,:]=np.copy(sigma1)
                #和一次更新不一样这个有些参数是迭代的。如果只用relu,随着层数加深,会导致提取的特征某些值很大
                #造成eta迭代也会很大。甚至矩阵不可逆，因为eye()相对小
                    a1 = (a1 - mu1) / sigma1 
                    
                if self.use_norm==2:#采用第一批的
                    if t==0:
                        mu1 = a1.mean(0)
                        sigma1 = a1.std(0)
                        sigma1 = np.maximum(sigma1, 0.0001)  
                        mu[t,i,:]=np.copy(mu1)
                        sigma[t,i,:]=np.copy(sigma1)
                        a1 = (a1 - mu1) / sigma1
                    else:
                        a1 = (a1 - mu[0,i,:]) / sigma[0,i,:]
                        
                
                #active function
                a1=self.active_function(a1)
                #a1 = relu(a1)#多种
                #a1 = sigmoid(a1)

                a1_temp = np.concatenate((x_train, a1, np.ones((a1.shape[0], 1))), axis=1)
                
                f.append(np.copy(a1_temp))
                
                a_input = np.concatenate((x_train, a1), axis=1)
                    
            F.append(np.copy(f))
            YY.append(np.copy(y_train))
            
        #为fd延长一批数据
        if  self.fd_extra_update==0:
            pass
        if  self.fd_extra_update==1:
            pass
        if  self.fd_extra_update==2:
            extra_F=np.copy(f)
            #extra_Y=np.copy(y_train)
        if  self.fd_extra_update==3:
            extra_index=np.random.randint(0, n_sample, (int(np.floor(self.extra_times * extra_num)),))#可指定数量,默认1.0
            x_train=X[extra_index]
            #extra_Y=Y[extra_index]
            a_input=np.copy(x_train)
            extra_F=[]
            for i in range(self.L):
                if i==0:
                    a1 = np.dot(a_input, weights1) + bias1
                else:
                    a1 = np.dot(a_input, weights2_L[i-1]) + bias2_L[i-1]
            
                #normalize:
                if self.use_norm==0:
                    pass
                if self.use_norm==1:
                    
                    mu1 = a1.mean(0)
                    sigma1 = a1.std(0)
                    sigma1 = np.maximum(sigma1, 0.0001)
                    
                    a1 = (a1 - mu1) / sigma1
            
                if self.use_norm==2:
                    a1 = (a1 - mu[0,i,:]) / sigma[0,i,:]
                    
                a1 = self.active_function(a1)
                a1_temp = np.concatenate((x_train, a1, np.ones((a1.shape[0], 1))), axis=1)
                extra_F.append(np.copy(a1_temp))
                a_input = np.concatenate((x_train, a1), 1)
            
        if self.update_mode==False:
            #self.C=np.clip(self.C, 2**-8, 2**8)
            eta_ = np.eye(n_feature+self.N+1)*self.C
            theta_ = np.zeros((n_feature+self.N+1,n_class))
            
            eta[:] = np.copy(eta_);eta_fd[:] = np.copy(eta_);theta[:] = np.copy(theta_);theta_fd[:] = np.copy(theta_);eta_k[:] = np.copy(eta_)#0时刻各层的初始值
            new_eta = np.zeros_like(eta);new_eta_ = np.zeros_like(eta);new_eta_k = np.zeros_like(eta)
            Theta.append(np.copy(theta));Theta_fd.append(np.copy(theta_fd));Eta.append(np.copy(eta));Eta_fd.append(np.copy(eta_fd));#收集初始值
            
            for j in range(it-1):#共it-1个值
                for i in range(self.L):
                    _ = eta[i]@( F[j][i].T@ F[j][i]@ theta[i] - F[j][i].T@YY[j])
                    theta[i]-=_#ridge更新每一层的参数
                    ___ = F[j+1][i]@eta[i]@F[j+1][i].T
                    new_eta[i] = eta[i]-eta[i]@F[j+1][i].T@np.linalg.inv(np.eye(F[j+1][i].shape[0])+___)@F[j+1][i]@eta[i]
                    
                    if j==0:
                        new_eta_[i]=np.copy(eta[i])
                    else:
                        new_eta_[i] = new_eta_[i]-new_eta_[i]@F[j][i].T@np.linalg.inv(np.eye(F[j][i].shape[0])+F[j][i]@new_eta_[i]@F[j][i].T)@F[j][i]@new_eta_[i]
                    
                    ___k = (self.fdk)*F[j+1][i]@new_eta_[i]@(F[j+1][i]).T
                    new_eta_k[i] = new_eta_[i]-new_eta_[i]@((self.fdk)*F[j+1][i]).T@np.linalg.inv(np.eye(F[j+1][i].shape[0])+___k)@(F[j+1][i])@new_eta_[i]
                    
                    theta_fd[i]-=new_eta_k[i]@((self.fdk*F[j+1][i].T@ F[j+1][i]+(1-self.fdk)*F[j][i].T@F[j][i])@ theta_fd[i] - F[j][i].T@YY[j])
                
                Theta.append(np.copy(theta))#各时刻的值存储
                Theta_fd.append(np.copy(theta_fd))
                Eta_fd.append(np.copy(new_eta))
                eta=np.copy(new_eta)
            
            j =(it-1)
            for i in range(self.L):#ridge的最后一次更新。注意不能改变eta_fd否则扰乱了fd的更新
                _ = eta[i]@( F[j][i].T@ F[j][i]@ theta[i] - F[j][i].T@YY[j])
                theta[i]-=_#ridge更新每一层的参数
             
            Theta.append(np.copy(theta))
            
            self.rg_index=[0]+[F[i][0].shape[0] for i in range(it)]
            #fd_index=rg_index[:-1]
            
            if  self.fd_extra_update>=2:#ridge不可以再更新
                for i in range(self.L):
                    #_ = eta[i]@( F[it-1][i].T@ F[it-1][i]@ theta[i] - F[it-1][i].T@YY[it-1])
                    #theta[i]-=_#ridge更新
                    new_eta_[i] = new_eta_[i]-new_eta_[i]@F[it-1][i].T@np.linalg.inv(np.eye(F[it-1][i].shape[0])+F[it-1][i]@new_eta_[i]@F[it-1][i].T)@F[it-1][i]@new_eta_[i]
                    ___k = (self.fdk)*extra_F[i]@new_eta_[i]@(extra_F[i]).T
                    new_eta_k[i] = new_eta_[i]-new_eta_[i]@((self.fdk)*extra_F[i]).T@np.linalg.inv(np.eye(extra_F[i].shape[0])+___k)@(extra_F[i])@new_eta_[i]
                    theta_fd[i]-=new_eta_k[i]@((self.fdk*extra_F[i].T@ extra_F[i]+(1-self.fdk)*F[it-1][i].T@ F[it-1][i])@ theta_fd[i] - F[it-1][i].T@YY[it-1])
                    
                    
                    
                #Theta.append(np.copy(theta))
                Theta_fd.append(np.copy(theta_fd))
                Eta_fd.append(np.copy(new_eta))
                #eta=np.copy(new_eta)
                #fd_index.append(rg_index[-1])
        if self.update_mode==True:
        #第一批数据采用闭式解
            for i in range(self.L):
                theta_,eta_=self.l2weights(F[0][i], YY[0], self.C)
                theta[i]=np.copy(theta_)#收集当前时刻
                eta[i]=np.copy(eta_)
                
                theta__,eta__=self.l2weights_fd(F[0][i], F[1][i], YY[0], self.C)
                
                if self.fd_first_use_ridge:
                    theta_fd[i]=np.copy(theta_)
                    eta_fd[i]=np.copy(eta_)
                else:
                    theta_fd[i]=np.copy(theta__)
                    eta_fd[i]=np.copy(eta__)    
                    
                
            Theta.append(np.copy(theta))#收集t个时刻
            Eta.append(np.copy(eta))
            
            Theta_fd.append(np.copy(theta_fd))#收集t个时刻
            Eta_fd.append(np.copy(eta_fd))
            
            #对于ridge
            for j in range(1,it):
                for i in range(self.L):
                    eta_=eta[i]-eta[i]@F[j][i].T@np.linalg.inv(np.eye(F[j][i].shape[0])+F[j][i]@eta[i]@F[j][i].T)@F[j][i]@eta[i]
                    theta_=theta[i]+eta_@F[j][i].T@(YY[j]-F[j][i]@theta[i])
                    theta[i]=np.copy(theta_)#收集当前时刻
                    eta[i]=np.copy(eta_)
                Theta.append(np.copy(theta))#收集t个时刻
                Eta.append(np.copy(eta))
                
                
            #对于forward
            for j in range(1,it-1):#forward最后一个batch不更新
                for i in range(self.L):
                    '''if (j==1)and self.fd_first_use_ridge:#如果fd的第一批采用了ridge，则衔接fd过程中j=1批次未用于fd更新。但这样做了效果反而不好故去掉
                        Ff=np.concatenate((F[j][i], F[j+1][i]), 0)
                        eta__=eta_fd[i]-eta_fd[i]@Ff.T@np.linalg.inv(np.eye(Ff.shape[0])+Ff@eta_fd[i]@Ff.T)@Ff@eta_fd[i]
                        theta__=theta_fd[i]+eta__@(F[j][i].T@YY[j]-Ff.T@Ff@theta_fd[i])
                    else:'''
                    eta__=eta_fd[i]-eta_fd[i]@F[j+1][i].T@np.linalg.inv(np.eye(F[j+1][i].shape[0])+F[j+1][i]@eta_fd[i]@F[j+1][i].T)@F[j+1][i]@eta_fd[i]
                    theta__=theta_fd[i]+eta__@(F[j][i].T@YY[j]-F[j+1][i].T@F[j+1][i]@theta_fd[i])
                    theta_fd[i]=np.copy(theta__)#收集当前时刻
                    eta_fd[i]=np.copy(eta__)
                Theta_fd.append(np.copy(theta_fd))#收集t个时刻
                Eta_fd.append(np.copy(eta_fd))
                
            if  self.fd_extra_update>=2:
                for i in range(self.L):
                    eta__=eta_fd[i]-eta_fd[i]@extra_F[i].T@np.linalg.inv(np.eye(extra_F[i].shape[0])+extra_F[i]@eta_fd[i]@extra_F[i].T)@extra_F[i]@eta_fd[i]
                    theta__=theta_fd[i]+eta__@(F[-1][i].T@YY[-1]-extra_F[i].T@extra_F[i]@theta_fd[i])
                    theta_fd[i]=np.copy(theta__)#收集当前时刻
                    eta_fd[i]=np.copy(eta__)
                Theta_fd.append(np.copy(theta_fd))#收集t个时刻
                Eta_fd.append(np.copy(eta_fd))
         
        #训练特征整理
        FF=[]#L@all_samples*(feature+N+1)
        for i in range(self.L):
            fff=[]
            for j in range(it):
                ff=F[j][i]
                fff.append(np.copy(ff))
            fff=np.concatenate(fff,0)
            FF.append(np.copy(fff))#, dtype=object
        
        #ridge的训练集预测
        scores=[]
        train_acc=[]
        for i in range(self.L):
            if self.fd_extra_update==1:
                ffff=FF[i]@Theta[-2][i]  #theta[i]#Theta[-1][i]
            else:
                ffff=FF[i]@Theta[-1][i]
            ffff=ffff-np.tile(ffff.max(1),(ffff.shape[1],1)).T
            num=np.exp(ffff)
            dem=num.sum(1)
            scores.append(np.copy(num/np.tile(dem,(ffff.shape[1],1)).T))
        del ffff, num, dem
        #forward的训练集预测
        scores_fd=[]
        train_acc_fd=[]
        for i in range(self.L):
            ffff=FF[i]@Theta_fd[-1][i]#Theta_fd[-1][i]           
            ffff=ffff-np.tile(ffff.max(1),(ffff.shape[1],1)).T
            num=np.exp(ffff)
            dem=num.sum(1)
            scores_fd.append(np.copy(num/np.tile(dem,(ffff.shape[1],1)).T))
        
        #YYY=np.vstack(YY)
        for i in range(self.L):#训练投票采用哪种？这里训练集采用的是累积scores
            
            train_acc.append(np.mean(np.argmax(Y,1)==np.argmax(sum(scores[:(i+1)])[0:n_sample,:],1)))
            train_acc_fd.append(np.mean(np.argmax(Y,1)==np.argmax(sum(scores_fd[:(i+1)])[0:n_sample,:],1)))
            
        #print(train_acc)
        #print(train_acc_fd)
        
        
        self.w=weights2_L
        self.w.insert(0,weights1)
        self.b =bias2_L
        self.b.insert(0,bias1)
        self.class_num=n_class
        
        if self.fd_extra_update==1:
            self.beta=Theta[-2]
        else:
            self.beta=Theta[-1]
        self.beta_fd=Theta_fd[-1]
        
        self.theta_array=Theta
        self.theta_fd_array=Theta_fd
        
        self.mu=mu
        self.sigma=sigma
        train_time = time.perf_counter() - start_time
        
        return train_time,train_acc,train_acc_fd,x_val, y_val,x_te, y_te

    def predict(self, x_test, y_test, index_tr, index_te):
        rg_index=np.cumsum(np.array(self.rg_index))
        index_tr=np.concatenate(([0],index_tr))
        index_te=np.concatenate(([0],index_te))
        task_boundary=[0]#初始参数
        for j in range(1,index_tr.shape[0]-1):
            bdy=np.where(rg_index>=index_tr[j])[0][0]#记录任务完成时，对应参数的索引
            task_boundary.append(bdy)#这个参数要对前面所有任务执行检测
        task_boundary.append(-1)#最后全部任务完成后的参数就是参数列的最后一个
            
        result=[]
        result_fd=[]
        
        n_sample = x_test.shape[0]
        n_layer = self.L
        beta = self.beta
        beta_fd = self.beta_fd
        
        theta_array=np.array(self.theta_array)
        theta_fd_array=np.array(self.theta_fd_array)
        
        weights = self.w
        biases = self.b
        mu = self.mu
        sigma = self.sigma
        n_class = self.class_num
        
        start_time = time.perf_counter()
        
        prob_scores = np.zeros((n_layer,n_sample,self.class_num), dtype=np.float64)
        prob_scores_fd = np.zeros_like(prob_scores)
        
        #每个时刻各层预测值的平均做总预测
        ir_loss_ridge=np.zeros((len(self.theta_array),self.L,n_sample,n_class)); 
        ir_loss_forward=np.zeros((len(self.theta_fd_array),self.L,n_sample,n_class))
        
        ir_loss_ridge_err=np.zeros((len(self.theta_array),self.L,n_sample,n_class)); 
        ir_loss_forward_err=np.zeros((len(self.theta_fd_array),self.L,n_sample,n_class))
        #每个时刻各层独立预测值。这里不是各层叠加
        ir_loss_ridge_sp=np.zeros((len(self.theta_array),self.L)); ir_loss_forward_sp=np.zeros((len(self.theta_fd_array),self.L))
            
        a_input = np.copy(x_test)
        
        for i in range(n_layer):
            w = np.copy(weights[i])
            b = np.copy(biases[i])

            a1 = np.dot(a_input, w) + b
            
            if self.use_norm==0:
                pass
            if self.use_norm==1:
                mu1 = a1.mean(0)
                sigma1 = a1.std(0)
                sigma1 = np.maximum(sigma1, 0.0001)  # for numerical stability
                a1 = (a1 - mu1) / sigma1
            if self.use_norm==2:
                a1 = (a1 - mu[0,i,:]) / sigma[0,i,:]
            

            a1=self.active_function(a1)
            #a1 = relu(a1)
            #a1 = sigmoid(a1)

            a1_temp = np.concatenate((x_test, a1, np.ones((n_sample, 1))), axis=1)

            
            beta1 = beta[i]#ridge regression第一层
            beta2 = beta_fd[i]#forward regression第一层
            
            #测试集不改变，beta是时间序列，结果表明性能随时间提升
            kk=a1_temp[np.newaxis,:,:]@np.squeeze(np.array(theta_array)[:,i,:,:])
            ir_loss_ridge_sp[:,i]=np.copy(np.mean(np.argmax(y_test,1)[np.newaxis,:]==np.argmax(kk,-1),-1))
            ir_loss_ridge_err[:,i,:,:]=np.copy(kk)
            kk=kk-np.max(kk,axis=-1,keepdims=True)
            num=np.exp(kk)
            dem=num.sum(-1)[:,:,np.newaxis]
            ir_loss_ridge[:,i,:,:]=np.copy(num/np.tile(dem,(1,1,kk.shape[-1])))
            
            kkk=a1_temp[np.newaxis,:,:]@np.squeeze(np.array(theta_fd_array)[:,i,:,:])
            ir_loss_forward_sp[:,i]=np.copy(np.mean(np.argmax(y_test,1)[np.newaxis,:]==np.argmax(kkk,-1),-1))
            ir_loss_forward_err[:,i,:,:]=np.copy(kkk)
            kkk=kkk-np.max(kkk,axis=-1,keepdims=True)
            num=np.exp(kkk)
            dem=num.sum(-1)[:,:,np.newaxis]
            ir_loss_forward[:,i,:,:]=np.copy(num/np.tile(dem,(1,1,kkk.shape[-1])))
                                
                
            #for ridge最终的beta
            y_test_temp = a1_temp.dot(beta1)
            y_test_temp1 = y_test_temp - np.tile(y_test_temp.max(1), (n_class, 1)).transpose()#每个样本预测值减去最大值
            num = np.exp(y_test_temp1)
            dem = num.sum(1)
            prob_scores_temp = num / np.tile(dem, (n_class, 1)).transpose()
            prob_scores[i] = np.copy(prob_scores_temp)
            
            #for forward最终的beta
            y_test_temp_fd = a1_temp.dot(beta2)
            y_test_temp1_fd = y_test_temp_fd - np.tile(y_test_temp_fd.max(1), (n_class, 1)).transpose()#每个样本预测值减去最大值
            num_fd = np.exp(y_test_temp1_fd)
            dem_fd = num_fd.sum(1)
            prob_scores_temp_fd = num_fd / np.tile(dem_fd, (n_class, 1)).transpose()
            prob_scores_fd[i] = np.copy(prob_scores_temp_fd)
            
            #for acc
            task_feature=[]
            for j in range(index_te.shape[0]-1):            
                task_feature.append(a1_temp[index_te[j]:index_te[j+1],:])#各任务的数据划分
            for j in range(len(task_feature)):
                result_=task_feature[j][np.newaxis,:,:]@np.array(theta_array[task_boundary[j+1:],i,:,:])
                result_=result_-np.tile(result_.max(-1)[:,:,np.newaxis],(1,1,n_class))
                num=np.exp(result_)
                dem=num.sum(-1)[:,:,np.newaxis]
                result_=num/np.tile(dem,(1,1,n_class))
                result.append(result_[:])
            
            for j in range(len(task_feature)):
                result__=task_feature[j][np.newaxis,:,:]@np.array(theta_fd_array[task_boundary[j+1:],i,:,:])
                result__=result__-np.tile(result__.max(-1)[:,:,np.newaxis],(1,1,n_class))
                num=np.exp(result__)
                dem=num.sum(-1)[:,:,np.newaxis]
                result__=num/np.tile(dem,(1,1,n_class))
                result_fd.append(result__[:])
            
            a_input = np.concatenate((x_test, a1), axis=1)
        #平均准确率
        self.IR_ridge_te=np.mean(np.argmax(y_test,1)[np.newaxis,:]==np.argmax(np.sum(ir_loss_ridge,1),-1),-1)
        self.IR_forward_te=np.mean(np.argmax(y_test,1)[np.newaxis,:]==np.argmax(np.sum(ir_loss_forward,1),-1),-1)
        #分层准确率
        self.IR_ridge_sp_te=np.copy(ir_loss_ridge_sp)#np.mean(ir_loss_ridge,-1)
        self.IR_forward_sp_te=np.copy(ir_loss_forward_sp)#np.mean(ir_loss_forward, -1)
        #累积准确率
        self.IR_ridge_spp_te=np.mean(np.argmax(y_test,1)[np.newaxis,np.newaxis,:]==np.argmax(np.cumsum(ir_loss_ridge,1),-1),-1)
        self.IR_forward_spp_te=np.mean(np.argmax(y_test,1)[np.newaxis,np.newaxis,:]==np.argmax(np.cumsum(ir_loss_forward,1),-1),-1)
        #平均误差
        #self.IR_ridge_err=np.sum((np.squeeze(np.mean(ir_loss_ridge_err,1))-y_test[np.newaxis,:])**2,axis=(1,2))#没有平均，太大
        #self.IR_ridge_err=np.sum(np.mean(np.squeeze(np.mean(ir_loss_ridge,1))-y_test[np.newaxis,:],1)**2,axis=1)
        self.IR_ridge_err=np.sum(np.mean(np.squeeze(np.mean(ir_loss_ridge,1))-y_test[np.newaxis,:],1)**2,axis=1)#取成平均误差
        self.IR_forward_err=np.sum(np.mean(np.squeeze(np.mean(ir_loss_forward,1))-y_test[np.newaxis,:],1)**2,axis=1)#先平均再平方
        kl_ridge=np.mean(np.sum(np.maximum(y_test,1e-10)[np.newaxis,:]*np.log(np.maximum(y_test,1e-10)[np.newaxis,:]/np.maximum(np.squeeze(np.mean(ir_loss_ridge,1)),1e-6)), axis=-1), axis=1)
        kl_fd=np.mean(np.sum(np.maximum(y_test,1e-10)[np.newaxis,:]*np.log(np.maximum(y_test,1e-10)[np.newaxis,:]/np.maximum(np.squeeze(np.mean(ir_loss_forward,1)),1e-6)), axis=-1), axis=1)
        
        test_time = time.perf_counter() - start_time
        #acc,采用各层softmax然后累加平均最大score
        Result=[]
        for j in range(len(task_feature)):
            Result.append( np.argmax(np.sum(np.array([result[j+len(task_feature)*jj] for jj in range(n_layer)]),0) ,-1)[:])#result记录了每一层的每个任务。现在，对每个任务，在每层的预测进行相加
        
        Result_fd=[]
        for j in range(len(task_feature)):
            Result_fd.append( np.argmax(np.sum(np.array([result_fd[j+len(task_feature)*jj] for jj in range(n_layer)]),0) ,-1)[:])
        
        Result=[np.mean(np.argmax(y_test[index_te[j]:index_te[j+1],:],-1)[np.newaxis,:]==Result[j],-1) for j in range(len(task_feature))][:]
        Result_fd=[np.mean(np.argmax(y_test[index_te[j]:index_te[j+1],:],-1)[np.newaxis,:]==Result_fd[j],-1) for j in range(len(task_feature))][:]
        return prob_scores, prob_scores_fd, self.error_times,self.IR_ridge_te, self.IR_forward_te,self.IR_ridge_sp_te,self.IR_forward_sp_te,\
    self.IR_ridge_spp_te,self.IR_forward_spp_te,self.IR_ridge_err,self.IR_forward_err,Result,Result_fd,task_boundary,kl_ridge,kl_fd

    def active_function(self, x):
        c=self.active_func
        if c==0:#relu
            y=np.maximum(x, 0)
        elif c==1:#sigmoid
            y= 1 / (1 + np.exp(-1 * (x)))
        elif c==2:#selu
            alpha=1.6732632423543772848170429916717
            scale=1.0507009873554804934193349852946
            x = np.clip(x, -200, 200)
            y=scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))
        elif c==3:
            y=np.tanh(x)
        elif c==4:
            alpha=0.01 ##leaky relu
            y=np.where(x>0, x, alpha * x)
        elif c==5:#radbas
            y=np.exp(-x**2)
        elif c==6:#硬极限
            y=np.where(x>=0, 1, 0)
        elif c==7:#三角波
            y=1 - np.abs(x - np.round(x))
        elif c==8:#swish y=x * (1/(1+np.exp(-x)))
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=RuntimeWarning)
                try:
                    y=x * (1/(1+np.exp(-x)))
                except RuntimeWarning as e:
                    print('\033[91m*****activate function: swish. May using clip to avoid overflow.*****'+'\033[0m')
                    print('\033[91m'+str(e)+'\033[0m\n')
                    self.error_times = True
                    self.print_info.append(str(e))
                    x=x/np.max(x)
                    y=x * (1/(1+np.exp(-x)))
                    #y=x * (1/(1+np.clip( np.exp(-x) , 0, np.exp(15))))#np.clip,-10,10
             
        return y
    def l2weights(self,x, y, c):
        
        [n_sample, n_feature] = x.shape
        if c>=2**20:#要改
            K=np.linalg.inv((np.eye(n_feature) / c + x.transpose().dot(x)))
            try:
                beta = np.linalg.inv(x.transpose().dot(x))@x.transpose()@y
            except:
                beta = np.linalg.pinv(x.transpose().dot(x))@x.transpose()@y
        else:
    
            #if n_feature < n_sample:#
            
            if (n_feature >= n_sample) and self.dual_solution ==True :#fd可以用对偶且满足特征总数>样本数
                try:#处理故障,c过小导致第一项很大
                    with warnings.catch_warnings():
                        warnings.filterwarnings("error", category=RuntimeWarning)
                        try:#处理警告,c过大造成第一项趋于0
                            self.solution_info.append('ridge dual 0')
                            K=np.linalg.inv((np.eye(n_feature) / c + x.transpose().dot(x)))
                            beta = x.transpose().dot(np.linalg.inv(np.eye(n_sample) / c + x.dot(x.transpose()))).dot(y)
                        except RuntimeWarning as e:
                            self.solution_info.append('ridge dual 1')
                            print('\033[91m#############Warning Caught--ridge reg-fea>>[abnormal]>>samp#############\033[0m')
                            print('\033[91m'+str(e)+'\033[0m')
                            self.error_times = True
                            self.print_info.append('abnormal-warning-ridge-fea>samp')
                            self.print_info.append(str(e))
                            K=np.linalg.inv(np.eye(n_feature) / 2**-2 + x.transpose().dot(x))
                            beta = x.transpose().dot(np.linalg.inv(np.eye(n_sample) / 2**-2 + x.dot(x.transpose()))).dot(y)
                            
                except Exception as e:
                    self.solution_info.append('ridge dual 2')
                    print('\033[91m'+str(e)+'\033[0m')
                    print("\033[91mridge reg [error] c="+str(c)+'\033[0m')
                    self.error_times = True
                    self.print_info.append('error-ridge-fea>samp')
                    self.print_info.append(str(e))
                    self.print_info.append('c='+str(c))
                      
                    K=np.linalg.inv((np.eye(n_feature) / 2**4 + x.transpose().dot(x)))
                    beta = x.transpose().dot(np.linalg.inv(np.eye(n_sample) / 2**4 + x.dot(x.transpose()))).dot(y)
            else:
                try:#处理故障,c过大导致第一项很小，奇异
                    with warnings.catch_warnings():
                        warnings.filterwarnings("error", category=RuntimeWarning)
                        try:#处理警告,c过小造成第一项很大，除0警告
                            self.solution_info.append('ridge prl 0')
                            K=np.linalg.inv((np.eye(n_feature) / c + x.transpose().dot(x)))
                            beta = K.dot(x.transpose()).dot(y)
                        except RuntimeWarning as e:
                            self.solution_info.append('ridge prl 1')
                            print('\033[91m#############Warning Caught--ridge reg-fea<<[abnormal]<<samp#############\033[0m')
                            print('\033[91m'+str(e)+'\033[0m')
                            self.error_times = True
                            self.print_info.append('abnormal-warning-ridge-fea<samp')
                            self.print_info.append(str(e))
                            K=np.linalg.inv(np.eye(n_feature) / 2**-2 + x.transpose().dot(x))
                            beta = K.dot(x.transpose()).dot(y)
                            
                except Exception as e:
                  self.solution_info.append('ridge prl 2')
                  print('\033[91m'+str(e)+'\033[0m')
                  print("\033[91mridge reg [error] c="+str(c)+'\033[0m')
                  self.error_times = True
                  self.print_info.append('error-ridge-fea<samp')
                  self.print_info.append(str(e))
                  self.print_info.append('c='+str(c))
                  
                  K=np.linalg.inv((np.eye(n_feature) / 2**4 + x.transpose().dot(x)))
                  beta = K.dot(x.transpose()).dot(y)
        
        return beta,K

    def l2weights_fd(self, x, xn, y, c):
        
        [n_sample, n_feature] = x.shape
        [nn_sample, nn_feature] = xn.shape
        
        if c>=2**20:#要改
            K=np.linalg.inv((np.eye(n_feature) / c + x.transpose().dot(x) + xn.transpose().dot(xn)))
            #beta = np.linalg.pinv(np.concatenate((x,xn),0).T@np.concatenate((x,xn),0))@y
            try:
                beta = np.linalg.inv((x.transpose().dot(x) + xn.transpose().dot(xn))).dot(x.transpose()).dot(y)
            except:
                beta = np.linalg.pinv((x.transpose().dot(x) + xn.transpose().dot(xn))).dot(x.transpose()).dot(y)
        else:
            if (n_feature >= n_sample) and self.dual_solution == True:
                #self.fd_first_use_ridge = False#开启了对偶解允许，表明ridge和fd都采用对偶解，所以fd第一次也要用对偶解
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("error", category=RuntimeWarning)
                        try:
                            self.solution_info.append('forward dual 0')
                            K=np.linalg.inv((np.eye(n_feature) / c + x.transpose().dot(x) + xn.transpose().dot(xn)))
                            beta = x.transpose().dot(np.linalg.inv(np.eye(n_sample) / c + x.dot(x.transpose()) + xn.dot(xn.transpose()))).dot(y)
                        except RuntimeWarning as e:
                            self.solution_info.append('forward dual 1')
                            print('\033[91m#############Warning Caught--forward reg-fea>>[abnormal]>>samp#############\033[0m')
                            print('\033[91m'+str(e)+'\033[0m')
                            self.error_times = True
                            self.print_info.append('abnormal-warning-forward-fea>samp')
                            self.print_info.append(str(e))
                            K=np.linalg.inv((np.eye(n_feature) / 2**-2 + x.transpose().dot(x) + xn.transpose().dot(xn)))
                            beta = x.transpose().dot(np.linalg.inv(np.eye(n_sample) / 2**-2 + x.dot(x.transpose()) + xn.dot(xn.transpose()))).dot(y)
                            
                except Exception as e:
                    self.solution_info.append('forward dual 2')
                    print('\033[91m'+str(e)+'\033[0m')
                    print("\033[91mforward reg [error] c="+str(c)+'\033[0m')
                    self.error_times = True
                    self.print_info.append('error-forward-fea>samp')
                    self.print_info.append(str(e))
                    self.print_info.append('c='+str(c))
                    K=np.linalg.inv((np.eye(n_feature) / 2**4 + x.transpose().dot(x) + xn.transpose().dot(xn)))
                    beta = x.transpose().dot(np.linalg.inv(np.eye(n_sample) / 2**4 + x.dot(x.transpose()) + xn.dot(xn.transpose()))).dot(y)
                    
            else:#n_feature < n_sample:
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("error", category=RuntimeWarning)
                        try:#处理警告,c过大造成第一项趋于0
                            self.solution_info.append('forward prl 0')
                            K=np.linalg.inv((np.eye(n_feature) / c + x.transpose().dot(x) + xn.transpose().dot(xn)))
                            beta = K.dot(x.transpose()).dot(y)
                        except RuntimeWarning as e:
                            self.solution_info.append('forward prl 1')
                            print('\033[91m#############Warning Caught--forward reg#############\033[0m')
                            print('\033[91m'+str(e)+'\033[0m')
                            self.error_times = True
                            self.print_info.append('abnormal-warning-forward-fea<samp')
                            self.print_info.append(str(e))
                            K=np.linalg.pinv((np.eye(n_feature) / 2**-2 + x.transpose().dot(x) + xn.transpose().dot(xn)))#
                            beta = K.dot(x.transpose()).dot(y)
                except Exception as e:
                  self.solution_info.append('forward prl 2')
                  print('\033[91m'+str(e)+'\033[0m')
                  print("\033[91mc="+str(c)+'\033[0m')
                  self.error_times = True
                  self.print_info.append('error-forward-fea<samp')
                  self.print_info.append(str(e))
                  self.print_info.append('c='+str(c))
                  
                  K=np.linalg.inv((np.eye(n_feature) / 2**4 + x.transpose().dot(x) + xn.transpose().dot(xn)))
                  beta = K.dot(x.transpose()).dot(y)
        
           
                '''KK=np.linalg.inv(np.eye(n_sample) / c + x.dot(x.transpose()))
                KKK=KK-KK@xn.dot(xn.transpose()).T@np.linalg.inv(np.eye(xn.shape[0])+xn.dot(xn.transpose())@@xn.dot(xn.transpose()).T)
                beta = x.transpose().dot(np.linalg.inv(np.eye(n_sample) / c + x.dot(x.transpose()) + xn.dot(xn.transpose()))).dot(y)'''
        
        return beta,K
    
    def xavier_init(self, n_input, n_output):
        limit = np.sqrt(6.0 / (n_input + n_output))
        return np.random.uniform(-limit, limit, size=(n_input, n_output))
    
    def kaiming_init(self, n_input, n_output):
        limit = np.sqrt(2.0 / n_input)
        return np.random.normal(0, limit, size=(n_input, n_output))
    
    def print_infoo(self):
        #print(str(self.solution_info))
        return self.error_times, self.print_info, self.solution_info

    '''
    def relu(x):
        y = np.maximum(x, 0)
        return y
    def sigmoid(x):
        y = 1 / (1 + np.exp(-1 * (x)))
        return y
    '''

    def calculate_acc(self, prob_scores, y_test):
        voting = self.voting
        n_scores = len(prob_scores)
        
        sum_prob_scores = 0
        classifier_voting_resp=np.zeros((y_test.shape[0], n_scores), dtype=int)#分别投票
        classifier_acc=[]#
        sum_scores_voting=np.zeros((y_test.shape[0], n_scores), dtype=int)#score累加投票
    
        i=0
        
        correct_idx = np.argmax(y_test, 1)
        for scores in prob_scores:
            classifier_voting_resp[:,i] = np.argmax(scores, 1)
            
            sum_prob_scores = sum_prob_scores + scores
            classifier_acc.append(np.mean(correct_idx == np.argmax(sum_prob_scores, 1)))
            
            sum_scores_voting[:,i] = np.argmax(sum_prob_scores, 1)
            i=i+1
        mean_prob_scores = sum_prob_scores / n_scores
    
        acc = np.mean(correct_idx == np.argmax(mean_prob_scores, 1))#和classifier_acc最后一个值相等
    
        classifier_voting_resp_acc = np.mean(correct_idx ==np.array([np.argmax(np.bincount(row)) for row in classifier_voting_resp]))
        sum_scores_voting_acc = np.mean(correct_idx ==np.array([np.argmax(np.bincount(row)) for row in sum_scores_voting]))
        return acc, classifier_acc, classifier_voting_resp_acc, sum_scores_voting_acc#分别投票,累积scores投票


class UCIDataset:
    def __init__(self, dataset, parent="E:/电脑/gdRVFL/junda_code/DLoader/UCIdata"):
        
        self.name = dataset
        self.root = Path(parent) / dataset
        data_file = sorted(self.root.glob(f'{dataset}*.dat'))[0]
        label_file = sorted(self.root.glob('label*.dat'))[0]
        val_file = sorted(self.root.glob('validation*.dat'))[0]
        fold_index = sorted(self.root.glob('folds*.dat'))[0]
        self.dataX = np.loadtxt(data_file, delimiter=',')#全部数据
        self.dataY = np.loadtxt(label_file, delimiter=',')#全部标签单列
        self.validation = np.loadtxt(val_file, delimiter=',')#验证集
        self.folds_index = np.loadtxt(fold_index, delimiter=',')#数据集折划分
        self.n_CV = self.folds_index.shape[1]#折数
        types = np.unique(self.dataY)#标签集合
        self.n_types = types.size#几分类
        # One hot coding for the target
        self.dataY_tmp = np.zeros((self.dataY.size, self.n_types))
        for i in range(self.n_types):
            for j in range(self.dataY.size):  # remove this loop
                if self.dataY[j] == types[i]:
                    self.dataY_tmp[j, i] = 1

    def getitem(self, CV):
        full_train_idx = np.where(self.folds_index[:, CV] == 0)[0]#第k折标记为0的3133个,包含验证集
        train_idx = np.where((self.folds_index[:, CV] == 0) & (self.validation[:, CV] == 0))[0]#第k折标记为0的 以及 验证集第k折标记为0的2526个
        test_idx = np.where(self.folds_index[:, CV] == 1)[0]#第k折标记为1的1044个
        val_idx = np.where(self.validation[:, CV] == 1)[0]#607个
        trainX = self.dataX[train_idx, :]
        trainY = self.dataY_tmp[train_idx, :]#纯训练
        testX = self.dataX[test_idx, :]
        testY = self.dataY_tmp[test_idx, :]#测试
        evalX = self.dataX[val_idx, :]
        evalY = self.dataY_tmp[val_idx, :]#验证
        full_train_x = self.dataX[full_train_idx, :]
        full_train_y = self.dataY_tmp[full_train_idx, :]#训练+验证
        

        return trainX, trainY, evalX, evalY, testX, testY, full_train_x, full_train_y##纯训练2532,验证601,测试1044,训练+验证3133
    def cil(self, TN, seed1, seed2, trainX, trainY, evalX, evalY, testX, testY):
        trX=[];trY=[];  vaX=[];vaY=[];   teX=[];teY=[]
        for i in range(self.n_types):
            index_tr=trainY[:,i]==1
            index_va=evalY[:,i]==1
            index_te=testY[:,i]==1
            
            trX.append(trainX[index_tr][:])
            trY.append(trainY[index_tr][:])
            
            vaX.append(evalX[index_va][:])
            vaY.append(evalY[index_va][:])
            
            teX.append(testX[index_te][:])
            teY.append(testY[index_te][:])
        
       
        np.random.seed(seed1)
        indices=np.random.permutation(self.n_types)#打乱任务顺序
        print('任务序列：'+str(indices))
        trX=[trX[i] for i in indices]
        trY=[trY[i] for i in indices]
        
        vaX=[vaX[i] for i in indices]
        vaY=[vaY[i] for i in indices]
        
        teX=[teX[i] for i in indices]
        teY=[teY[i] for i in indices]
        
        
        class_per_task=self.n_types//TN#每个任务多少个类，类总数/任务数
        tr_X=[];tr_Y=[];  va_X=[];va_Y=[];   te_X=[];te_Y=[]
        tr_num=[];va_num=[];te_num=[];
        
        for i in range(TN):
            if i==TN-1:
                #order=task_orders[i*class_per_task:]
                tr_X.append(self.rand_perm(seed2, i, np.concatenate(trX[i*class_per_task:],0) ) )
                tr_Y.append(self.rand_perm(seed2, i, np.concatenate(trY[i*class_per_task:],0) ) )
                tr_num.append(tr_X[-1].shape[0])
                
                #va_X.append(self.rand_perm(seed2, i+10000, np.concatenate(vaX[i*class_per_task:],0) ) )
                #va_Y.append(self.rand_perm(seed2, i+10000, np.concatenate(vaY[i*class_per_task:],0) ) )
                va_X.append(np.concatenate(vaX[i*class_per_task:],0) ) 
                va_Y.append(np.concatenate(vaY[i*class_per_task:],0) ) 
                va_num.append(va_X[-1].shape[0])
                
                #te_X.append(self.rand_perm(seed2, i+20000, np.concatenate(teX[i*class_per_task:],0) ) )
                #te_Y.append(self.rand_perm(seed2, i+20000, np.concatenate(teY[i*class_per_task:],0) ) )
                te_X.append(np.concatenate(teX[i*class_per_task:],0) ) 
                te_Y.append(np.concatenate(teY[i*class_per_task:],0) ) 
                te_num.append(te_X[-1].shape[0])
            else:
                #order=task_orders[i*class_per_task:(i+1)*class_per_task]
                tr_X.append(self.rand_perm(seed2, i, np.concatenate(trX[i*class_per_task:(i+1)*class_per_task],0) ) )
                tr_Y.append(self.rand_perm(seed2, i, np.concatenate(trY[i*class_per_task:(i+1)*class_per_task],0) ) )
                tr_num.append(tr_X[-1].shape[0])
            
                #va_X.append(self.rand_perm(seed2, i+10000, np.concatenate(vaX[i*class_per_task:(i+1)*class_per_task],0) ) )
                #va_Y.append(self.rand_perm(seed2, i+10000, np.concatenate(vaY[i*class_per_task:(i+1)*class_per_task],0) ) )
                va_X.append(np.concatenate(vaX[i*class_per_task:(i+1)*class_per_task],0) ) 
                va_Y.append(np.concatenate(vaY[i*class_per_task:(i+1)*class_per_task],0) ) 
                va_num.append(va_X[-1].shape[0])
                
                #te_X.append(self.rand_perm(seed2, i+20000, np.concatenate(teX[i*class_per_task:(i+1)*class_per_task],0) ) )
                #te_Y.append(self.rand_perm(seed2, i+20000, np.concatenate(teY[i*class_per_task:(i+1)*class_per_task],0) ) )
                te_X.append(np.concatenate(teX[i*class_per_task:(i+1)*class_per_task],0) ) 
                te_Y.append(np.concatenate(teY[i*class_per_task:(i+1)*class_per_task],0) ) 
                te_num.append(te_X[-1].shape[0])
                
        return tr_X, tr_Y, va_X, va_Y, te_X, te_Y, indices, np.cumsum(np.array(tr_num)), np.cumsum(np.array(va_num)), np.cumsum(np.array(te_num))
    def rand_perm(self, seed2, index, x):
        if seed2<0:
            xx=x#不打乱顺序直接输出
        else:
            np.random.seed(seed2+index)
            perm=np.random.permutation(x.shape[0])
        
            xx=x[perm][:]
        
        return xx

        
if __name__ == "__main__" :
    import pickle as pkl
    loader = UCIDataset('letter', parent="/UCIdata")
    print(f'折数：{loader.n_CV}')#折数
    
    split_idx=0
    X_train, y_train, X_val, y_val , X_te , y_te , _ , _  = loader.getitem(split_idx)#折数
    train_xx, train_yy , val_x, val_y , test_x , test_y, indices,tr_num,va_num,te_num=loader.cil( loader.n_types, 0+split_idx, 0+split_idx+100, X_train, y_train, X_val, y_val , X_te , y_te)

    assert tr_num.shape[0]==te_num.shape[0]
    train_x=np.concatenate(train_xx,0)
    train_y=np.concatenate(train_yy,0)
    val_x=np.concatenate(val_x,0)
    val_y=np.concatenate(val_y,0)
    test_x=np.concatenate(test_x,0)
    test_y=np.concatenate(test_y,0)
    
    
    '''
    #简单实现Class-IL
    index=train_yy==1
    train_x=train_xx[index[:,2]];train_y=train_yy[index[:,2]];
    train_x=np.concatenate((train_x,train_xx[index[:,1]]),0);train_y=np.concatenate((train_y,train_yy[index[:,1]]),0)
    train_x=np.concatenate((train_x,train_xx[index[:,0]]),0);train_y=np.concatenate((train_y,train_yy[index[:,0]]),0)
    '''
    batch_por=0.02
    if 0:#E:/电脑/k forward learning/hyper-params/ridge_best_1718241977.0197752.pk
        with open("/home/wangjunda/CL/hyperparam_tunning/classification/letter/edRVFL_ridge_classification/best_1718241977.0197752.pk" , "rb") as file :
            params1 = pkl.load(file)['params']
            
        net_seed=42
            
        model_temp = edRVFL_ridge_classification(L=params1['L'], N=params1['N'], C=params1['C'], scale_w=1.0, scale_b=0.2, init_por=0, batch_por=batch_por, \
                            data_norm=params1['data_norm'], add_val=False,  use_uniform=params1['use_uniform'], use_norm=params1['use_norm'], active_func=params1['active_func'], fd_first_use_ridge=False, fd_extra_update=params1['fd_extra_update'],\
                                voting=0, seed=net_seed, dual_solution=False, update_mode=False, fdk=1)
            #
        train_time,train_acc,train_acc_fd,x_val, y_val,x_te, y_te = model_temp.train(train_x, train_y , val_x, val_y , test_x , test_y)# Y 为1/-1编码
        scores, scores_fd, error_times,IR_ridge_te, IR_forward_te,IR_ridge_sp_te,IR_forward_sp_te,\
            IR_ridge_spp_te,IR_forward_spp_te,IR_ridge_err,IR_forward_err,Result,Result_fd,task_boundary,kl_ridge,kl_fd = model_temp.predict(x_te, y_te, tr_num, te_num)
            
        test_acc, classifier_acc,voting_resp_acc, sum_scores_voting_acc = model_temp.calculate_acc(scores, y_te)
        test_acc_fd, classifier_acc_fd,voting_resp_acc_fd, sum_scores_voting_acc_fd = model_temp.calculate_acc(scores_fd, y_te)
        
        print(train_acc)
        print(train_acc_fd)
        print('test acc:'+str(test_acc))
        print(classifier_acc)
        print(voting_resp_acc)
        print(sum_scores_voting_acc)
        print('test fd acc:'+str(test_acc_fd))
        print(classifier_acc_fd)
        print(voting_resp_acc_fd)
        print(sum_scores_voting_acc_fd)
        iserror, print_info, solution_info=model_temp.print_infoo()
        print(iserror, print_info[1:], solution_info[1:])
    
    
        with open(f'/home/wangjunda/CL/hyperparam_tunning/classification/letter/rg_fd_letter_{batch_por}.pkl', 'wb') as file:
            pkl.dump([IR_ridge_te, IR_forward_te,IR_ridge_sp_te,IR_forward_sp_te,\
                IR_ridge_spp_te,IR_forward_spp_te,IR_ridge_err,IR_forward_err,Result,Result_fd,task_boundary,\
                    test_acc, classifier_acc,voting_resp_acc, sum_scores_voting_acc,\
                        test_acc_fd, classifier_acc_fd,voting_resp_acc_fd, sum_scores_voting_acc_fd,kl_ridge,kl_fd], file)
        del model_temp
        
    if 0:
        with open("/home/wangjunda/CL/hyperparam_tunning/classification/letter/edRVFL_forward_classification/best_1718241977.0197752.pk" , "rb") as file :
            params2 = pkl.load(file)['params']
    
        net_seed=42
            
        model_temp2 = edRVFL_ridge_classification(L=params2['L'], N=params2['N'], C=params2['C'], scale_w=1.0, scale_b=0.2, init_por=0, batch_por=batch_por, \
                            data_norm=params2['data_norm'], add_val=False,  use_uniform=params2['use_uniform'], use_norm=params2['use_norm'], active_func=params2['active_func'], fd_first_use_ridge=False, fd_extra_update=params2['fd_extra_update'],\
                                voting=0, seed=net_seed, dual_solution=False, update_mode=False, fdk=params2['fdk'])
            #
        train_time,train_acc,train_acc_fd,x_val, y_val,x_te, y_te = model_temp2.train(train_x, train_y , val_x, val_y , test_x , test_y)# Y 为1/-1编码
        scores, scores_fd, error_times,IR_ridge_te, IR_forward_te,IR_ridge_sp_te,IR_forward_sp_te,\
            IR_ridge_spp_te,IR_forward_spp_te,IR_ridge_err,IR_forward_err,Result,Result_fd,task_boundary,kl_ridge,kl_fd = model_temp2.predict(x_te, y_te, tr_num, te_num)
            
        test_acc, classifier_acc,voting_resp_acc, sum_scores_voting_acc = model_temp2.calculate_acc(scores, y_te)
        test_acc_fd, classifier_acc_fd,voting_resp_acc_fd, sum_scores_voting_acc_fd = model_temp2.calculate_acc(scores_fd, y_te)
        
        print(train_acc)
        print(train_acc_fd)
        print('test acc:'+str(test_acc))
        print(classifier_acc)
        print(voting_resp_acc)
        print(sum_scores_voting_acc)
        print('test fd acc:'+str(test_acc_fd))
        print(classifier_acc_fd)
        print(voting_resp_acc_fd)
        print(sum_scores_voting_acc_fd)
        iserror, print_info, solution_info=model_temp2.print_infoo()
        print(iserror, print_info[1:], solution_info[1:])
        
        with open(f'/home/wangjunda/CL/hyperparam_tunning/classification/letter/kfd_letter_{batch_por}.pkl', 'wb') as file2:
            pkl.dump([IR_ridge_te, IR_forward_te,IR_ridge_sp_te,IR_forward_sp_te,\
                IR_ridge_spp_te,IR_forward_spp_te,IR_ridge_err,IR_forward_err,Result,Result_fd,task_boundary,\
                    test_acc, classifier_acc,voting_resp_acc, sum_scores_voting_acc,\
                        test_acc_fd, classifier_acc_fd,voting_resp_acc_fd, sum_scores_voting_acc_fd,kl_ridge,kl_fd], file2)
    
    
    from edRVFLkk_CLASS_evaluate import edRVFL_ridge_bayes_classification
    if 0:
        with open("/home/wangjunda/CL/hyperparam_tunning/classification/letter/edRVFL_bayes_classification/best_1718241977.0197752.pk" , "rb") as file :
            params3 = pkl.load(file)['params']
    
        net_seed=42
            
        model_temp3 = edRVFL_ridge_bayes_classification(L=params3['L'], N=params3['N'], C=params3['C'], scale_w=1.0, scale_b=0.2, init_por=0, batch_por=batch_por, \
                            data_norm=params3['data_norm'], add_val=False,  use_uniform=params3['use_uniform'], use_norm=params3['use_norm'], active_func=params3['active_func'], fd_first_use_ridge=False, fd_extra_update=params3['fd_extra_update'],\
                                voting=0, seed=net_seed, dual_solution=False, update_mode=False, bayes_error=params3["bayes_error"] ,bayes_shrinkage=params3["bayes_shrinkage"])
            #
        train_time,train_acc,train_acc_fd,x_val, y_val,x_te, y_te,KT1,KT = model_temp3.train(train_x, train_y , val_x, val_y , test_x , test_y)# Y 为1/-1编码
        scores, scores_fd, error_times,IR_ridge_te, IR_forward_te,IR_ridge_sp_te,IR_forward_sp_te,\
            IR_ridge_spp_te,IR_forward_spp_te,IR_ridge_err,IR_forward_err,Result,Result_fd,task_boundary,kl_ridge,kl_fd = model_temp3.predict(x_te, y_te, tr_num, te_num)
            
        test_acc, classifier_acc,voting_resp_acc, sum_scores_voting_acc = model_temp3.calculate_acc(scores, y_te)
        test_acc_fd, classifier_acc_fd,voting_resp_acc_fd, sum_scores_voting_acc_fd = model_temp3.calculate_acc(scores_fd, y_te)
        
        print(train_acc)
        print(train_acc_fd)
        print('test acc:'+str(test_acc))
        print(classifier_acc)
        print(voting_resp_acc)
        print(sum_scores_voting_acc)
        print('test fd acc:'+str(test_acc_fd))
        print(classifier_acc_fd)
        print(voting_resp_acc_fd)
        print(sum_scores_voting_acc_fd)
        iserror, print_info, solution_info=model_temp3.print_infoo()
        print(iserror, print_info[1:], solution_info[1:])
        
        with open(f'/home/wangjunda/CL/hyperparam_tunning/classification/letter/bayes_letter_{batch_por}.pkl', 'wb') as file3:
            pkl.dump([IR_ridge_te, IR_forward_te,IR_ridge_sp_te,IR_forward_sp_te,\
                IR_ridge_spp_te,IR_forward_spp_te,IR_ridge_err,IR_forward_err,Result,Result_fd,[task_boundary,KT1,KT],\
                    test_acc, classifier_acc,voting_resp_acc, sum_scores_voting_acc,\
                        test_acc_fd, classifier_acc_fd,voting_resp_acc_fd, sum_scores_voting_acc_fd,kl_ridge,kl_fd], file3)
    
    
    if 1:
        
        print('train set:'+str(train_x.shape[0])+' validate set:'+str(val_x.shape[0])+' test set:'+str(test_x.shape[0]))
        
        model_temp4 = edRVFL_ridge_bayes_classification(L=3, N=256, C=4, scale_w=1.0, scale_b=0.2, init_por=0.04, batch_por=0.02, \
                            data_norm=2, add_val=False,  use_uniform=0, use_norm=0, active_func=1, fd_first_use_ridge=False, fd_extra_update=3,\
                                voting=0, seed=1, dual_solution=False, update_mode=False, bayes_error=0.001, bayes_shrinkage=1)#shrinkage的搜索范围是回归的c倍(适当变大)，c是类别数
            #
        train_time,train_acc,train_acc_fd,x_val, y_val,x_te, y_te,KT1,KT = model_temp4.train(train_x, train_y , val_x, val_y , test_x , test_y)# Y 为1/-1编码
        scores, scores_fd, error_times,IR_ridge_te, IR_forward_te,IR_ridge_sp_te,IR_forward_sp_te,\
            IR_ridge_spp_te,IR_forward_spp_te,IR_ridge_err,IR_forward_err,Result,Result_fd,task_boundary,kl_ridge,kl_fd= model_temp4.predict(x_te, y_te, tr_num, te_num)
        test_acc, classifier_acc,voting_resp_acc, sum_scores_voting_acc = model_temp4.calculate_acc(scores, y_te)
        test_acc_fd, classifier_acc_fd,voting_resp_acc_fd, sum_scores_voting_acc_fd = model_temp4.calculate_acc(scores_fd, y_te)
        
        print(train_acc)
        print(train_acc_fd)
        print('test acc:'+str(test_acc))
        print(classifier_acc)
        print(voting_resp_acc)
        print(sum_scores_voting_acc)
        print('test fd acc:'+str(test_acc_fd))
        print(classifier_acc_fd)
        print(voting_resp_acc_fd)
        print(sum_scores_voting_acc_fd)
        iserror, print_info, solution_info=model_temp.print_infoo()
        print(iserror, print_info[1:], solution_info[1:])
    
    
    
