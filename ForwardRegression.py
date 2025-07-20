#!/usr/bin/env python
# coding: utf-8



import sys
import numpy as np
from scipy.stats import t as t_law
import matplotlib.pyplot as plt
plt.style.use("ggplot")
folder = "xps/RidgeForwardComparison/"

# Uncomment to check regularization effect
# lamdas = [1/T, 1/np.log(T), 1, 10]
np.random.seed(0)

#训练数据的产生也可以修改：分布，分布参数. 单样本用正态分布生成数据，噪声也用正态分布
n_iters = 10#运行次数
T = 1000#序列长度
sigma0 =2 #噪声比重，重要
sigma1 =1
# index_reg=3
d = 12  #
#lamdas = [0.005,0.01,0.02, 1/T] ##1/T, 1/np.log(T),, 10
lamdas = [0.2, 0.5,0.8,1.0,1.5,2.0,3.0]
#lamda = 0.005
theta_star = np.random.multivariate_normal([6],[[2]],d).reshape(-1,) #2*np.random.rand(d)-1
methods = ['Ridge','Forward','k0-Forward','k1-Forward','k2-Forward','k3-Forward','bayes-Forward','k4-Forward','k5-Forward','k6-Forward','Oracle']  # 'Oracle' is using the true parameter
fdk=[0.2,0.4,0.6,0.8,1.2,1.5,2.0]
kappa=0.9#重要

mean_online_regrets_reg = np.zeros((len(lamdas), len(methods)-1, T))
std_online_regrets_reg = np.zeros((len(lamdas), len(methods)-1, T))


colors = ['#377eb8', '#ff7f00', '#4daf4a','#f781bf', '#a65628', '#984ea3','#999999', '#e41a1c', '#dede00']
s_colors = ['#377eb8', '#ff7f00', '#4daf4a','#f781bf', '#a65628', '#984ea3','#999999', '#e41a1c', '#dede00']
online_loss = np.zeros((len(lamdas), n_iters, len(methods), T))
fst_terms = np.zeros((len(lamdas), n_iters,len(methods),T))#n*2*(T)
scd_terms = np.zeros((len(lamdas), n_iters,len(methods),T))#n*1*(T-1)
noise_terms = np.zeros((len(lamdas), n_iters,  T))

theta_norm=[]
prediction=np.zeros((len(lamdas), n_iters, len(methods)+1, T))#预测值2个，oracle预测值（未加噪声的真值），加噪声的真值
noise_max=0
def target(x):
    return np.dot(x,theta_star)
print(theta_star)

A1=[];A2=[]
# Run
for lam in range(len(lamdas)):
    for it in range(n_iters):  # iterations to average the effect of noise
        thetas = np.zeros((len(methods),d))#更新的θ
        thetas[-1]=theta_star#最后一行是oracle
        eta = np.eye(d)/lamdas[lam]
        x_train = np.random.normal(0,1,(T,d))#  rand(T,d) 产生随机数据
        y_train = target(x_train)#标签
        noise = sigma0*np.random.normal(0,1,(T,))+sigma1# randn(T)  噪声
        noise_terms[lam, it] = noise 
        y_train += noise  #加噪
        X, Y = np.max(np.abs(x_train)), max(abs(y_train))  # max([max(np.abs(x)) for x in x_train])np.linalg.norm(x)最大观测
        for t in range(T):
            x_t, y_t = x_train[t], y_train[t]
            online_loss[lam, it, :, t] = (np.dot(thetas,x_t)-y_t)**2        #是否应除以2，不除2：4Y^2；除2：2Y^2。下面fst,scd_terms要一起改
            prediction[lam, it, :,t]=np.append(np.dot(thetas,x_t), y_t)
            _ = np.dot(eta,np.dot(np.outer(x_t,x_t),thetas[0])-y_t*x_t)  # ridge更新
            __ = np.inner(x_t,np.dot(eta,x_t))  #
            
            if t<T-1:
                ___ = np.inner(x_train[t+1],np.dot(eta,x_train[t+1]))  #ridge
                if t==0:
                    __k0=np.inner(x_t,np.dot(eta,x_t))
                    ___k0=np.inner(x_train[t+1],np.dot(eta,x_train[t+1]))
                    __k1=np.inner(x_t,np.dot(eta,x_t))
                    ___k1=np.inner(x_train[t+1],np.dot(eta,x_train[t+1]))
                    __k2=np.inner(x_t,np.dot(eta,x_t))
                    ___k2=np.inner(x_train[t+1],np.dot(eta,x_train[t+1]))
                    __k3=np.inner(x_t,np.dot(eta,x_t))
                    ___k3=np.inner(x_train[t+1],np.dot(eta,x_train[t+1]))
                    __k4=np.inner(x_t,np.dot(eta,x_t))
                    ___k4=np.inner(x_train[t+1],np.dot(eta,x_train[t+1]))
                    __k5=np.inner(x_t,np.dot(eta,x_t))
                    ___k5=np.inner(x_train[t+1],np.dot(eta,x_train[t+1]))
                    __k6=np.inner(x_t,np.dot(eta,x_t))
                    ___k6=np.inner(x_train[t+1],np.dot(eta,x_train[t+1]))
                    __kb=np.inner(x_t,np.dot(eta,x_t))
                    ___kb=np.inner(x_train[t+1],np.dot(eta,x_train[t+1]))
                else:
                    __k0=np.inner(x_t,np.dot(new_eta_k0,x_t))
                    ___k0=np.inner(x_train[t+1],np.dot(new_eta_k0,x_train[t+1]))
                    __k1=np.inner(x_t,np.dot(new_eta_k1,x_t))
                    ___k1=np.inner(x_train[t+1],np.dot(new_eta_k1,x_train[t+1]))
                    __k2=np.inner(x_t,np.dot(new_eta_k2,x_t))
                    ___k2=np.inner(x_train[t+1],np.dot(new_eta_k2,x_train[t+1]))
                    __k3=np.inner(x_t,np.dot(new_eta_k3,x_t))
                    ___k3=np.inner(x_train[t+1],np.dot(new_eta_k3,x_train[t+1]))
                    __k4=np.inner(x_t,np.dot(new_eta_k4,x_t))
                    ___k4=np.inner(x_train[t+1],np.dot(new_eta_k4,x_train[t+1]))
                    __k5=np.inner(x_t,np.dot(new_eta_k5,x_t))
                    ___k5=np.inner(x_train[t+1],np.dot(new_eta_k5,x_train[t+1]))
                    __k6=np.inner(x_t,np.dot(new_eta_k6,x_t))
                    ___k6=np.inner(x_train[t+1],np.dot(new_eta_k6,x_train[t+1]))
                    __kb=np.inner(x_t,np.dot(new_eta_kb,x_t))
                    ___kb=np.inner(x_train[t+1],np.dot(new_eta_kb,x_train[t+1]))
                    
                new_eta = eta - np.dot(eta,np.dot(np.outer(x_train[t+1],x_train[t+1]),eta))/(1+___)  #ridge
                if t==0:
                    new_eta_=np.copy(eta)
                    byk=kappa*np.inner(x_t,np.dot(eta,x_t))
                else:
                    new_eta_ = new_eta_-np.dot(new_eta_,np.dot(np.outer(x_t,x_t),new_eta_))/(1+np.inner(x_t,np.dot(new_eta_,x_t)))
                    byk=kappa*np.inner(x_t,np.dot(new_eta_,x_t));A1.append(byk)#有错的原版
                    #byk=kappa*np.inner(x_t,np.dot(new_eta_kb,x_t));A2.append(byk)#正确版,但这里效果不好
                    
                new_eta_k0=new_eta_-np.dot(new_eta_,np.dot((fdk[0])*np.outer(x_train[t+1],x_train[t+1]),new_eta_))/(1+(fdk[0])*np.inner(x_train[t+1],np.dot(new_eta_,x_train[t+1])))
                new_eta_k1=new_eta_-np.dot(new_eta_,np.dot((fdk[1])*np.outer(x_train[t+1],x_train[t+1]),new_eta_))/(1+(fdk[1])*np.inner(x_train[t+1],np.dot(new_eta_,x_train[t+1])))
                new_eta_k2=new_eta_-np.dot(new_eta_,np.dot((fdk[2])*np.outer(x_train[t+1],x_train[t+1]),new_eta_))/(1+(fdk[2])*np.inner(x_train[t+1],np.dot(new_eta_,x_train[t+1])))
                new_eta_k3=new_eta_-np.dot(new_eta_,np.dot((fdk[3])*np.outer(x_train[t+1],x_train[t+1]),new_eta_))/(1+(fdk[3])*np.inner(x_train[t+1],np.dot(new_eta_,x_train[t+1])))
                new_eta_k4=new_eta_-np.dot(new_eta_,np.dot((fdk[4])*np.outer(x_train[t+1],x_train[t+1]),new_eta_))/(1+(fdk[4])*np.inner(x_train[t+1],np.dot(new_eta_,x_train[t+1])))
                new_eta_k5=new_eta_-np.dot(new_eta_,np.dot((fdk[5])*np.outer(x_train[t+1],x_train[t+1]),new_eta_))/(1+(fdk[5])*np.inner(x_train[t+1],np.dot(new_eta_,x_train[t+1])))
                new_eta_k6=new_eta_-np.dot(new_eta_,np.dot((fdk[6])*np.outer(x_train[t+1],x_train[t+1]),new_eta_))/(1+(fdk[6])*np.inner(x_train[t+1],np.dot(new_eta_,x_train[t+1])))
                new_eta_kb=new_eta_-np.dot(new_eta_,np.dot((byk)*np.outer(x_train[t+1],x_train[t+1]),new_eta_))/(1+(byk)*np.inner(x_train[t+1],np.dot(new_eta_,x_train[t+1])))
            thetas[0] -= _  # theta is updated for Ridge reg
            fst_terms[lam, it, 0, t] = online_loss[lam, it, 0, t]*__  # 
            if t<T-1:
                thetas[1] -= np.dot(new_eta,np.dot(np.outer(x_train[t+1],x_train[t+1]),thetas[1])-y_t*x_t)  # Forward algo
                thetas[2]-=np.dot(new_eta_k0,np.dot(fdk[0]*np.outer(x_train[t+1],x_train[t+1])+(1-fdk[0])*np.outer(x_t,x_t),thetas[2])-y_t*x_t)
                thetas[3]-=np.dot(new_eta_k1,np.dot(fdk[1]*np.outer(x_train[t+1],x_train[t+1])+(1-fdk[1])*np.outer(x_t,x_t),thetas[3])-y_t*x_t)
                thetas[4]-=np.dot(new_eta_k2,np.dot(fdk[2]*np.outer(x_train[t+1],x_train[t+1])+(1-fdk[2])*np.outer(x_t,x_t),thetas[4])-y_t*x_t)
                thetas[5]-=np.dot(new_eta_k3,np.dot(fdk[3]*np.outer(x_train[t+1],x_train[t+1])+(1-fdk[3])*np.outer(x_t,x_t),thetas[5])-y_t*x_t)
                thetas[7]-=np.dot(new_eta_k4,np.dot(fdk[4]*np.outer(x_train[t+1],x_train[t+1])+(1-fdk[4])*np.outer(x_t,x_t),thetas[7])-y_t*x_t)
                thetas[8]-=np.dot(new_eta_k5,np.dot(fdk[5]*np.outer(x_train[t+1],x_train[t+1])+(1-fdk[5])*np.outer(x_t,x_t),thetas[8])-y_t*x_t)
                thetas[9]-=np.dot(new_eta_k6,np.dot(fdk[6]*np.outer(x_train[t+1],x_train[t+1])+(1-fdk[6])*np.outer(x_t,x_t),thetas[9])-y_t*x_t)
                thetas[6]-=np.dot(new_eta_kb,np.dot(byk*np.outer(x_train[t+1],x_train[t+1])+(1-byk)*np.outer(x_t,x_t),thetas[6])-y_t*x_t)

                scd_terms[lam, it, 1, t] = (np.inner(thetas[1],x_train[t+1])**2)*___  #是否应除以2
                scd_terms[lam, it, 2, t] = (np.inner(thetas[2],x_train[t+1])**2)*___k0*fdk[0]*fdk[0]
                scd_terms[lam, it, 3, t] = (np.inner(thetas[3],x_train[t+1])**2)*___k1*fdk[1]*fdk[1]
                scd_terms[lam, it, 4, t] = (np.inner(thetas[4],x_train[t+1])**2)*___k2*fdk[2]*fdk[2]
                scd_terms[lam, it, 5, t] = (np.inner(thetas[5],x_train[t+1])**2)*___k3*fdk[3]*fdk[3]
                scd_terms[lam, it, 7, t] = (np.inner(thetas[7],x_train[t+1])**2)*___k3*fdk[4]*fdk[4]
                scd_terms[lam, it, 8, t] = (np.inner(thetas[8],x_train[t+1])**2)*___k3*fdk[5]*fdk[5]
                scd_terms[lam, it, 9, t] = (np.inner(thetas[9],x_train[t+1])**2)*___k3*fdk[6]*fdk[6]
                scd_terms[lam, it, 6, t] = (np.inner(thetas[6],x_train[t+1])**2)*___kb*byk*byk
                
            fst_terms[lam, it, 1, t] = (y_t**2)*__   #是否应除以2
            fst_terms[lam, it, 2, t] = (y_t**2)*__k0*fdk[0]
            fst_terms[lam, it, 3, t] = (y_t**2)*__k1*fdk[1]
            fst_terms[lam, it, 4, t] = (y_t**2)*__k2*fdk[2]
            fst_terms[lam, it, 5, t] = (y_t**2)*__k3*fdk[3]
            fst_terms[lam, it, 7, t] = (y_t**2)*__k4*fdk[4]
            fst_terms[lam, it, 8, t] = (y_t**2)*__k5*fdk[5]
            fst_terms[lam, it, 9, t] = (y_t**2)*__k6*fdk[6]
            fst_terms[lam, it, 6, t] = (y_t**2)*__kb*byk
            eta = new_eta[:]

        theta_norm.append(np.copy(thetas))#每次试验的最终theta记录
        noise_max=max(noise_max,max(abs(noise)))#n次试验中，噪声的最大值
    # ### Data characteristics
    print('lambda=',lamdas[lam])#惩罚系数
    print("||theta|| is: ",round(np.linalg.norm(theta_star),2),end=" ")#theta真值的范数
    print(" and X is: ", round(X,2),end=" ")#最后一次iteration
    print(" and Y is: ",round(Y,2),end=" ")#最后一次iteration
    #print("max |eps_t| is: ",max(abs(noise)))#最后一次iteration
    print("max |eps_t| is: ",noise_max)#全部试验的最大值
    print(methods)
    theta_norm_=np.array(theta_norm)-theta_star[np.newaxis, np.newaxis, :]#求n次试验中的平均值
    theta_norm_=np.mean(np.sqrt(((theta_norm_)**2).sum(axis=2)), 0)
    theta_norm=[]

# ### Results: mean arrays
mean_online_loss = np.mean(online_loss, axis=1)#多次实验的loss均值, methods*T
mean_online_regrets_reg = mean_online_loss[:,:-1,:]-mean_online_loss[:,[-1],:]#减去噪声造成的损失

std_online_loss = np.std(online_loss[:,:,:-1,:]-online_loss[:,:,[-1],:], axis=1)#methods*T，多次实验的loss标准差，或者np.std(online_loss, axis=0)？
std_online_regrets_reg = std_online_loss[:,:,:]

mean_fst_terms = np.mean(fst_terms, axis=1)#methods*T
mean_scd_terms = np.mean(scd_terms, axis=1)
std_fst_terms = np.std(fst_terms, axis=1)
std_scd_terms = np.std(scd_terms, axis=1)
########################################################################################################################################################################################################################################

########################################################################################################################################################################################################################################

#Fig 1:ridge
i=0
plt.figure(0)
for lam in range(len(lamdas)):
    a=np.clip( (mean_online_loss[lam,i]-mean_online_loss[lam,-1]), 0.01, None)
    if lam==len(lamdas)-1:
        plt.plot(a, color=colors[lam], label='-R $k$=0'+' $\lambda$='+str(lamdas[lam]))#颜色，标注
        plt.plot(np.cumsum(a), color=colors[lam], linestyle='--')
    else:
        plt.plot(a, color=colors[lam], label='-R $k$=0'+' $\lambda$='+str(lamdas[lam]))#颜色，标注
        plt.plot(np.cumsum(a), color=colors[lam], linestyle='--')
    b=np.clip( ((mean_online_loss[lam,i]-mean_online_loss[lam,-1])+np.abs(std_online_loss[lam,i])), 0.01, None)
    c=np.clip( ((mean_online_loss[lam,i]-mean_online_loss[lam,-1])-np.abs(std_online_loss[lam,i])), 0.01, None)
    #plt.fill_between(np.arange(T),b,c, color=s_colors[lam],alpha=0.15)   #一致颜色+透明度
    #np.cumsumnp.cumsum np.cumsumnp.cumsum
    plt.xscale('log')
    plt.yscale('log')
plt.legend()    
plt.xlabel('Time $t$')
plt.ylabel('Immediate Regret')
#plt.title('Online Regret of Single Data Ridge Learning')
ylim=plt.gca().get_ylim()
plt.show()

########################################################################################################################################################################################################################################
''''''
#Fig 2:forward
i=1
plt.figure(1)
for lam in range(len(lamdas)):
    a=np.clip( mean_online_loss[lam,i]-mean_online_loss[lam,-1], 0.01, None)
    if lam==len(lamdas)-1:
        plt.plot(a, color=colors[lam], label='-F $k$=1'+' $\lambda$='+str(lamdas[lam]))#颜色，标注
        plt.plot(np.cumsum(a), color=colors[lam], linestyle='--')
    else:
        plt.plot(a, color=colors[lam], label='-F $k$=1'+' $\lambda$='+str(lamdas[lam]))#颜色，标注
        plt.plot(np.cumsum(a), color=colors[lam], linestyle='--')
    
    b=np.clip( (mean_online_loss[lam,i]-mean_online_loss[lam,-1])+np.abs(std_online_loss[lam,i]), 0.01, None)
    c=np.clip( (mean_online_loss[lam,i]-mean_online_loss[lam,-1])-np.abs(std_online_loss[lam,i]), 0.01, None)
    #plt.fill_between(np.arange(T),b,c, color=s_colors[lam],alpha=0.2)   #一致颜色+透明度
    #np.cumsumnp.cumsum np.cumsumnp.cumsum
    plt.xscale('log')
    plt.yscale('log')
plt.legend()    
plt.xlabel('Time $t$')
plt.ylabel('Immediate Regret')
#plt.title('Online Regret of Single Data Forward Learning')
plt.ylim(ylim)
plt.show()
#sys.exit()
########################################################################################################################################################################################################################################
''''''
#Fig 2:forward
i=2
plt.figure(2)
for lam in range(len(lamdas)):
    a=np.clip( mean_online_loss[lam,i]-mean_online_loss[lam,-1], 0.01, None)
    if lam==len(lamdas)-1:
        plt.plot(a, color=colors[lam], label='-$k$F $k$=0.2'+' $\lambda$='+str(lamdas[lam]))#颜色，标注
        plt.plot(np.cumsum(a), color=colors[lam], linestyle='--')
    else:
        plt.plot(a, color=colors[lam], label='-$k$F $k$=0.2'+' $\lambda$='+str(lamdas[lam]))#颜色，标注
        plt.plot(np.cumsum(a), color=colors[lam], linestyle='--')
    
    b=np.clip( (mean_online_loss[lam,i]-mean_online_loss[lam,-1])+np.abs(std_online_loss[lam,i]), 0.01, None)
    c=np.clip( (mean_online_loss[lam,i]-mean_online_loss[lam,-1])-np.abs(std_online_loss[lam,i]), 0.01, None)
    #plt.fill_between(np.arange(T),b,c, color=s_colors[lam],alpha=0.2)   #一致颜色+透明度
    #np.cumsumnp.cumsum np.cumsumnp.cumsum
    plt.xscale('log')
    plt.yscale('log')
plt.legend()    
plt.xlabel('Time $t$')
plt.ylabel('Immediate Regret')
#plt.title('Online Regret of Single Data Forward Learning')
plt.ylim(ylim)
plt.show()

########################################################################################################################################################################################################################################
''''''
#Fig 2:forward
i=3
plt.figure(3)
for lam in range(len(lamdas)):
    a=np.clip( mean_online_loss[lam,i]-mean_online_loss[lam,-1], 0.01, None)
    if lam==len(lamdas)-1:
        plt.plot(a, color=colors[lam], label='-$k$F $k$=0.4'+' $\lambda$='+str(lamdas[lam]))#颜色，标注
        plt.plot(np.cumsum(a), color=colors[lam], linestyle='--')
    else:
        plt.plot(a, color=colors[lam], label='-$k$F $k$=0.4'+' $\lambda$='+str(lamdas[lam]))#颜色，标注
        plt.plot(np.cumsum(a), color=colors[lam], linestyle='--')
    
    b=np.clip( (mean_online_loss[lam,i]-mean_online_loss[lam,-1])+np.abs(std_online_loss[lam,i]), 0.01, None)
    c=np.clip( (mean_online_loss[lam,i]-mean_online_loss[lam,-1])-np.abs(std_online_loss[lam,i]), 0.01, None)
    #plt.fill_between(np.arange(T),b,c, color=s_colors[lam],alpha=0.2)   #一致颜色+透明度
    #np.cumsumnp.cumsum np.cumsumnp.cumsum
    plt.xscale('log')
    plt.yscale('log')
plt.legend()    
plt.xlabel('Time $t$')
plt.ylabel('Immediate Regret')
#plt.title('Online Regret of Single Data Forward Learning')
plt.ylim(ylim)
plt.show()

########################################################################################################################################################################################################################################
''''''
#Fig 2:forward
i=4
plt.figure(4)
for lam in range(len(lamdas)):
    a=np.clip( mean_online_loss[lam,i]-mean_online_loss[lam,-1], 0.01, None)
    if lam==len(lamdas)-1:
        plt.plot(a, color=colors[lam], label='-$k$F $k$=0.6'+' $\lambda$='+str(lamdas[lam]))#颜色，标注
        plt.plot(np.cumsum(a), color=colors[lam], linestyle='--')
    else:
        plt.plot(a, color=colors[lam], label='-$k$F $k$=0.6'+' $\lambda$='+str(lamdas[lam]))#颜色，标注
        plt.plot(np.cumsum(a), color=colors[lam], linestyle='--')
    
    b=np.clip( (mean_online_loss[lam,i]-mean_online_loss[lam,-1])+np.abs(std_online_loss[lam,i]), 0.01, None)
    c=np.clip( (mean_online_loss[lam,i]-mean_online_loss[lam,-1])-np.abs(std_online_loss[lam,i]), 0.01, None)
    #plt.fill_between(np.arange(T),b,c, color=s_colors[lam],alpha=0.2)   #一致颜色+透明度
    #np.cumsumnp.cumsum np.cumsumnp.cumsum
    plt.xscale('log')
    plt.yscale('log')
plt.legend()    
plt.xlabel('Time $t$')
plt.ylabel('Immediate Regret')
#plt.title('Online Regret of Single Data Forward Learning')
plt.ylim(ylim)
plt.show()

########################################################################################################################################################################################################################################
''''''
#Fig 2:forward
i=5
plt.figure(5)
for lam in range(len(lamdas)):
    a=np.clip( mean_online_loss[lam,i]-mean_online_loss[lam,-1], 0.01, None)
    if lam==len(lamdas)-1:
        plt.plot(a, color=colors[lam], label='-$k$F $k$=0.8'+' $\lambda$='+str(lamdas[lam]))#颜色，标注
        plt.plot(np.cumsum(a), color=colors[lam], linestyle='--')
    else:
        plt.plot(a, color=colors[lam], label='-$k$F $k$=0.8'+' $\lambda$='+str(lamdas[lam]))#颜色，标注
        plt.plot(np.cumsum(a), color=colors[lam], linestyle='--')
    
    b=np.clip( (mean_online_loss[lam,i]-mean_online_loss[lam,-1])+np.abs(std_online_loss[lam,i]), 0.01, None)
    c=np.clip( (mean_online_loss[lam,i]-mean_online_loss[lam,-1])-np.abs(std_online_loss[lam,i]), 0.01, None)
    #plt.fill_between(np.arange(T),b,c, color=s_colors[lam],alpha=0.2)   #一致颜色+透明度
    #np.cumsumnp.cumsum np.cumsumnp.cumsum
    plt.xscale('log')
    plt.yscale('log')
plt.legend()    
plt.xlabel('Time $t$')
plt.ylabel('Immediate Regret')
#plt.title('Online Regret of Single Data Forward Learning')
plt.ylim(ylim)
plt.show()

########################################################################################################################################################################################################################################
''''''
#Fig 2:forward
i=6
plt.figure(6)
for lam in range(len(lamdas)):
    a=np.clip( mean_online_loss[lam,i]-mean_online_loss[lam,-1], 0.01, None)
    if lam==len(lamdas)-1:
        plt.plot(a, color=colors[lam], label='-$k$F-$Bayes$'+' $\lambda$='+str(lamdas[lam]))#颜色，标注
        plt.plot(np.cumsum(a), color=colors[lam], linestyle='--')
    else:
        plt.plot(a, color=colors[lam], label='-$k$F-$Bayes$'+' $\lambda$='+str(lamdas[lam]))#颜色，标注
        plt.plot(np.cumsum(a), color=colors[lam], linestyle='--')
    
    b=np.clip( (mean_online_loss[lam,i]-mean_online_loss[lam,-1])+np.abs(std_online_loss[lam,i]), 0.01, None)
    c=np.clip( (mean_online_loss[lam,i]-mean_online_loss[lam,-1])-np.abs(std_online_loss[lam,i]), 0.01, None)
    #plt.fill_between(np.arange(T),b,c, color=s_colors[lam],alpha=0.2)   #一致颜色+透明度
    #np.cumsumnp.cumsum np.cumsumnp.cumsum
    plt.xscale('log')
    plt.yscale('log')
plt.legend()    
plt.xlabel('Time $t$')
plt.ylabel('Immediate Regret')
#plt.title('Online Regret of Single Data Forward Learning')
plt.ylim(ylim)
plt.show()
########################################################################################################################################################################################################################################
''''''
#Fig 2:forward
i=7
plt.figure(7)
for lam in range(len(lamdas)):
    a=np.clip( mean_online_loss[lam,i]-mean_online_loss[lam,-1], 0.01, None)
    if lam==len(lamdas)-1:
        plt.plot(a, color=colors[lam], label='-$k$F $k$=1.2'+' $\lambda$='+str(lamdas[lam]))#颜色，标注
        plt.plot(np.cumsum(a), color=colors[lam], linestyle='--')
    else:
        plt.plot(a, color=colors[lam], label='-$k$F $k$=1.2'+' $\lambda$='+str(lamdas[lam]))#颜色，标注
        plt.plot(np.cumsum(a), color=colors[lam], linestyle='--')
    
    b=np.clip( (mean_online_loss[lam,i]-mean_online_loss[lam,-1])+np.abs(std_online_loss[lam,i]), 0.01, None)
    c=np.clip( (mean_online_loss[lam,i]-mean_online_loss[lam,-1])-np.abs(std_online_loss[lam,i]), 0.01, None)
    #plt.fill_between(np.arange(T),b,c, color=s_colors[lam],alpha=0.2)   #一致颜色+透明度
    #np.cumsumnp.cumsum np.cumsumnp.cumsum
    plt.xscale('log')
    plt.yscale('log')
plt.legend()    
plt.xlabel('Time $t$')
plt.ylabel('Immediate Regret')
#plt.title('Online Regret of Single Data Forward Learning')
plt.ylim(ylim)
plt.show()
########################################################################################################################################################################################################################################
''''''
#Fig 2:forward
i=8
plt.figure(8)
for lam in range(len(lamdas)):
    a=np.clip( mean_online_loss[lam,i]-mean_online_loss[lam,-1], 0.01, None)
    if lam==len(lamdas)-1:
        plt.plot(a, color=colors[lam], label='-$k$F $k$=1.5'+' $\lambda$='+str(lamdas[lam]))#颜色，标注
        plt.plot(np.cumsum(a), color=colors[lam], linestyle='--')
    else:
        plt.plot(a, color=colors[lam], label='-$k$F $k$=1.5'+' $\lambda$='+str(lamdas[lam]))#颜色，标注
        plt.plot(np.cumsum(a), color=colors[lam], linestyle='--')
    
    b=np.clip( (mean_online_loss[lam,i]-mean_online_loss[lam,-1])+np.abs(std_online_loss[lam,i]), 0.01, None)
    c=np.clip( (mean_online_loss[lam,i]-mean_online_loss[lam,-1])-np.abs(std_online_loss[lam,i]), 0.01, None)
    #plt.fill_between(np.arange(T),b,c, color=s_colors[lam],alpha=0.2)   #一致颜色+透明度
    #np.cumsumnp.cumsum np.cumsumnp.cumsum
    plt.xscale('log')
    plt.yscale('log')
plt.legend()    
plt.xlabel('Time $t$')
plt.ylabel('Immediate Regret')
#plt.title('Online Regret of Single Data Forward Learning')
plt.ylim(ylim)
plt.show()
########################################################################################################################################################################################################################################
''''''
#Fig 2:forward
i=9
plt.figure(9)
for lam in range(len(lamdas)):
    a=np.clip( mean_online_loss[lam,i]-mean_online_loss[lam,-1], 0.01, None)
    if lam==len(lamdas)-1:
        plt.plot(a, color=colors[lam], label='-$k$F $k$=2.0'+' $\lambda$='+str(lamdas[lam]))#颜色，标注
        plt.plot(np.cumsum(a), color=colors[lam], linestyle='--')
    else:
        plt.plot(a, color=colors[lam], label='-$k$F $k$=2.0'+' $\lambda$='+str(lamdas[lam]))#颜色，标注
        plt.plot(np.cumsum(a), color=colors[lam], linestyle='--')
    
    b=np.clip( (mean_online_loss[lam,i]-mean_online_loss[lam,-1])+np.abs(std_online_loss[lam,i]), 0.01, None)
    c=np.clip( (mean_online_loss[lam,i]-mean_online_loss[lam,-1])-np.abs(std_online_loss[lam,i]), 0.01, None)
    #plt.fill_between(np.arange(T),b,c, color=s_colors[lam],alpha=0.2)   #一致颜色+透明度
    #np.cumsumnp.cumsum np.cumsumnp.cumsum
    plt.xscale('log')
    plt.yscale('log')
plt.legend()    
plt.xlabel('Time $t$')
plt.ylabel('Immediate Regret')
#plt.title('Online Regret of Single Data Forward Learning')
plt.ylim(ylim)
plt.show()
########################################################################################################################################################################################################################################
A=np.reshape(np.array(A1),(-1,998))
B=np.reshape(np.array(A),(-1,10,998))
C=np.mean(B,1)
#Fig 2:forward
plt.figure(66)
for i in range(C.shape[0]):
    plt.plot(C[i], color=colors[i], label='-$k$F-$Bayes$'+' $\lambda$='+str(lamdas[i]))#颜色，标注
        
plt.legend()    
plt.xlabel('Time $t$')
plt.ylabel('Adaptive $k$ Value')
#plt.title('Online Regret of Single Data Forward Learning')
plt.show()

######################################################################################################################################################################################
#Fig 3: loss terms
plt.figure(11)
i=0
styles = ['^', 'v', '<', '>', '.', ',', 'o', '*','D']
for lam in range(1):
    
    plt.plot(mean_fst_terms[lam,i], label='lambda='+str(lamdas[lam])+' '+methods[i]+' 1stTerm',color=colors[i],marker=styles[lam],markevery=200,linewidth=0.5)#每个λ采用三种颜色，不同λ采用不同style区分
        #plt.fill_between(np.arange(T)+1, mean_fst_terms[i]-std_fst_terms[i], mean_fst_terms[i]+std_fst_terms[i],color=s_colors[i],alpha=0.2)
    #a=mean_scd_terms[lam,1,:-1]
    #a=np.insert(a,0,a[0])
    #plt.plot(a, color=colors[2],label='lambda='+str(lamdas[lam])+' '+methods[1]+' 2ndTerm',marker=styles[lam],markevery=100,linewidth=0.5)#markersize=6
plt.legend()
plt.xlabel('Time $t$')
plt.ylabel('Immediate Regret Terms')
plt.yscale('log')
#plt.title('Online Regret Terms of Ridge and Forward Learning')
plt.show()    

########################################################################################################################################################################################################################################
plt.figure(12)
i=1
styles = ['^', 'v', '<', '>', '.', ',', 'o', '*','D']
for lam in range(1):
    
    plt.plot(mean_fst_terms[lam,i], label='lambda='+str(lamdas[lam])+' '+methods[i]+' 1stTerm',color=colors[i],marker=styles[lam],markevery=200,linewidth=0.5)#每个λ采用三种颜色，不同λ采用不同style区分
        #plt.fill_between(np.arange(T)+1, mean_fst_terms[i]-std_fst_terms[i], mean_fst_terms[i]+std_fst_terms[i],color=s_colors[i],alpha=0.2)
    a=mean_scd_terms[lam,i]
    #a=np.insert(a,0,a[0])
    plt.plot(a, color=colors[2],label='lambda='+str(lamdas[lam])+' '+methods[1]+' 2ndTerm',marker=styles[lam],markevery=100,linewidth=0.5)#markersize=6
plt.legend()
plt.xlabel('Time $t$')
plt.ylabel('Immediate Regret Terms')
plt.yscale('log')
#plt.title('Online Regret Terms of Ridge and Forward Learning')
plt.show()    

########################################################################################################################################################################################
plt.figure(13)
i=2
styles = ['^', 'v', '<', '>', '.', ',', 'o', '*','D']
for lam in range(1):
    
    plt.plot(mean_fst_terms[lam,i], label='lambda='+str(lamdas[lam])+' '+methods[i]+' 1stTerm',color=colors[i],marker=styles[lam],markevery=200,linewidth=0.5)#每个λ采用三种颜色，不同λ采用不同style区分
        #plt.fill_between(np.arange(T)+1, mean_fst_terms[i]-std_fst_terms[i], mean_fst_terms[i]+std_fst_terms[i],color=s_colors[i],alpha=0.2)
    a=mean_scd_terms[lam,i]
    #a=np.insert(a,0,a[0])
    plt.plot(a, color=colors[2],label='lambda='+str(lamdas[lam])+' '+methods[1]+' 2ndTerm',marker=styles[lam],markevery=100,linewidth=0.5)#markersize=6
plt.legend()
plt.xlabel('Time $t$')
plt.ylabel('Immediate Regret Terms')
plt.yscale('log')
#plt.title('Online Regret Terms of Ridge and Forward Learning')
plt.show()    

########################################################################################################################################################################################
plt.figure(14)
i=3
styles = ['^', 'v', '<', '>', '.', ',', 'o', '*','D']
for lam in range(1):
    
    plt.plot(mean_fst_terms[lam,i], label='lambda='+str(lamdas[lam])+' '+methods[i]+' 1stTerm',color=colors[i],marker=styles[lam],markevery=200,linewidth=0.5)#每个λ采用三种颜色，不同λ采用不同style区分
        #plt.fill_between(np.arange(T)+1, mean_fst_terms[i]-std_fst_terms[i], mean_fst_terms[i]+std_fst_terms[i],color=s_colors[i],alpha=0.2)
    a=mean_scd_terms[lam,i]
    #a=np.insert(a,0,a[0])
    plt.plot(a, color=colors[2],label='lambda='+str(lamdas[lam])+' '+methods[1]+' 2ndTerm',marker=styles[lam],markevery=100,linewidth=0.5)#markersize=6
plt.legend()
plt.xlabel('Time $t$')
plt.ylabel('Immediate Regret Terms')
plt.yscale('log')
#plt.title('Online Regret Terms of Ridge and Forward Learning')
plt.show()    

########################################################################################################################################################################################
plt.figure(15)
i=4
styles = ['^', 'v', '<', '>', '.', ',', 'o', '*','D']
for lam in range(1):
    
    plt.plot(mean_fst_terms[lam,i], label='lambda='+str(lamdas[lam])+' '+methods[i]+' 1stTerm',color=colors[i],marker=styles[lam],markevery=200,linewidth=0.5)#每个λ采用三种颜色，不同λ采用不同style区分
        #plt.fill_between(np.arange(T)+1, mean_fst_terms[i]-std_fst_terms[i], mean_fst_terms[i]+std_fst_terms[i],color=s_colors[i],alpha=0.2)
    a=mean_scd_terms[lam,i]
    #a=np.insert(a,0,a[0])
    plt.plot(a, color=colors[2],label='lambda='+str(lamdas[lam])+' '+methods[1]+' 2ndTerm',marker=styles[lam],markevery=100,linewidth=0.5)#markersize=6
plt.legend()
plt.xlabel('Time $t$')
plt.ylabel('Immediate Regret Terms')
plt.yscale('log')
#plt.title('Online Regret Terms of Ridge and Forward Learning')
plt.show()    

########################################################################################################################################################################################
plt.figure(16)
i=5
styles = ['^', 'v', '<', '>', '.', ',', 'o', '*','D']
for lam in range(1):
    
    plt.plot(mean_fst_terms[lam,i], label='lambda='+str(lamdas[lam])+' '+methods[i]+' 1stTerm',color=colors[i],marker=styles[lam],markevery=200,linewidth=0.5)#每个λ采用三种颜色，不同λ采用不同style区分
        #plt.fill_between(np.arange(T)+1, mean_fst_terms[i]-std_fst_terms[i], mean_fst_terms[i]+std_fst_terms[i],color=s_colors[i],alpha=0.2)
    a=mean_scd_terms[lam,i]
    #a=np.insert(a,0,a[0])
    plt.plot(a, color=colors[2],label='lambda='+str(lamdas[lam])+' '+methods[1]+' 2ndTerm',marker=styles[lam],markevery=100,linewidth=0.5)#markersize=6
plt.legend()
plt.xlabel('Time $t$')
plt.ylabel('Immediate Regret Terms')
plt.yscale('log')
#plt.title('Online Regret Terms of Ridge and Forward Learning')
plt.show()    

########################################################################################################################################################################################
plt.figure(17)
i=6
styles = ['^', 'v', '<', '>', '.', ',', 'o', '*','D']
for lam in range(1):
    
    plt.plot(mean_fst_terms[lam,i], label='lambda='+str(lamdas[lam])+' '+methods[i]+' 1stTerm',color=colors[i],marker=styles[lam],markevery=200,linewidth=0.5)#每个λ采用三种颜色，不同λ采用不同style区分
        #plt.fill_between(np.arange(T)+1, mean_fst_terms[i]-std_fst_terms[i], mean_fst_terms[i]+std_fst_terms[i],color=s_colors[i],alpha=0.2)
    a=mean_scd_terms[lam,i]
    #a=np.insert(a,0,a[0])
    plt.plot(a, color=colors[2],label='lambda='+str(lamdas[lam])+' '+methods[1]+' 2ndTerm',marker=styles[lam],markevery=100,linewidth=0.5)#markersize=6
plt.legend()
plt.xlabel('Time $t$')
plt.ylabel('Immediate Regret Terms')
plt.yscale('log')
#plt.title('Online Regret Terms of Ridge and Forward Learning')
plt.show()    

########################################################################################################################################################################################
'''
for lam in range(len(lamdas)):
    for i in range(len(methods)-1):
        a=np.clip( mean_online_loss[lam,i]-mean_online_loss[lam,-1], 0.01, None)
        plt.plot(a, color=colors[i],label=str(lamdas[lam])+methods[i])
        
        b=np.clip( (mean_online_loss[lam,i]-mean_online_loss[lam,-1])+np.abs(std_online_loss[lam,i]), 0.01, None)
        c=np.clip( (mean_online_loss[lam,i]-mean_online_loss[lam,-1])-np.abs(std_online_loss[lam,i]), 0.01, None)
        plt.fill_between(np.arange(T),b,c, color=s_colors[i],alpha=0.2)   
    #np.cumsumnp.cumsum np.cumsumnp.cumsum
    #     plt.plot(np.cumsum(mean_online_loss[i]+mean_scd_terms[i]),color=colors[i],linestyle='dotted')
        plt.yscale('log')

plt.legend()
title = "xps/OnlineRegret_ThetaNorm="+str(round(np.linalg.norm(theta_star),2))
'''


# ### Regularization effect
# Regularization effect on ridge regression
'''
lstyles = ['solid','dotted']
styles = ['o', '^', 's', 'D', 'p', 'v', '*']
for i in range(len(lamdas)):
    plt.plot(mean_online_regrets_reg[0],color=colors[i],label=str(lamdas[i]))
    plt.fill_between(np.arange(T),mean_online_regrets_reg[0]-std_online_regrets_reg[0],mean_online_regrets_reg[0]+std_online_regrets_reg[0], color=s_colors[i],alpha=0.2)   
#     plt.plot(np.cumsum(mean_online_loss[i]+mean_scd_terms[i]),color=colors[i],linestyle='dotted')
plt.yscale('log')     
plt.xscale('log')
plt.xlabel('#Observations')
plt.ylabel('Online Regret')
plt.legend()
title = folder+"Regularization_online_regret_ridge"
plt.show()

# Regularization effect on forward regression

lstyles = ['solid','dotted']
styles = ['o', '^', 's', 'D', 'p', 'v', '*']
for i in range(len(lamdas)):
    plt.plot(mean_online_regrets_reg[i,1],color=colors[i],label=str(lamdas[i]))
    plt.fill_between(np.arange(T),mean_online_regrets_reg[i,1]-std_online_regrets_reg[i,1],mean_online_regrets_reg[i,1]+std_online_regrets_reg[i,1], color=s_colors[i],alpha=0.2)   
#     plt.plot(np.cumsum(mean_online_loss[i]+mean_scd_terms[i]),color=colors[i],linestyle='dotted')
plt.yscale('log')  
plt.xscale('log')
plt.xlabel('#Observations')
plt.ylabel('Online Regret')
plt.legend()
title = folder+"Regularization_online_regret_forward"
plt.show()
'''

# ### Plot Regret Terms

# Here you can visualize what we call the first and second terms in the paper.






