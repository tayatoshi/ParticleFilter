
# coding: utf-8

# In[ ]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

pddata=pd.read_csv('data.csv')
print(pddata.head())
pddata.plot(figsize=(12,4))
plt.show()
data=np.array(pddata)


# ### 非線形・非ガウス状態空間モデル
# $
# \begin{align}
# x_{t}&=f_{t}(x_{t-1},\upsilon_{t})\\
# y_{t}&=h_{t}(x_{t},\omega_{t})
# \end{align}
# $

# In[ ]:

class ParticleFilter:
    def __init__(self,y,n_particle,upsilon2,omega2):
        self.y=y
        self.length=len(y)
        self.length_of_time=len(y)
        self.n_particle=n_particle
        self.upsilon2=upsilon2
        self.omega2=omega2
        self.filtered_value = np.zeros(self.length)
        print('OK!!')

    def init_particle(self):
        # x(i)_0|0
        particles = []
        predicts = []
        init=np.random.uniform(-2,2,self.n_particle)
        particles.append(init)
        predicts.append(init)
        return({'particles':particles,'predicts':predicts})

    def get_likelihood(self,ensemble,t):
        #今回は正規分布を仮定
        likelihoodes=(1/np.sqrt(2*np.pi*self.omega2))*np.exp((-1/(2*self.omega2))*((self.y[t]-ensemble[t])**2))
        return(likelihoodes)

    def one_predict(self,ensemble,t):
        # x(i)_t|t-1
        noise=np.random.normal(0,np.sqrt(self.upsilon2),self.n_particle)
        predict=ensemble[t]+noise
        return(predict)

    def filtering(self,ensemble,t):
        # x(i)_t|t
        likelihood=self.get_likelihood(ensemble,t)
        beta=likelihood/likelihood.sum()
        #print('beta',beta)
        filtering_value=np.sum(beta*ensemble[t])
        return({'beta':beta,'filtering_value':filtering_value})

    def resumpling(self,ensemble,weight):
        # sample=np.zeros(self.n_particle)
        # for i in range(self.n_particle):
            # sample[i]=np.random.choice(ensemble,p=weight)
        sample=np.random.choice(ensemble,p=weight,size=self.n_particle)
        return(sample)

    def simulate(self,seed=123):
        np.random.seed(seed)
        particles=self.init_particle()['particles']
        predicts=self.init_particle()['predicts']
        filtered_value=np.zeros(self.length)
        filtered_value[0]=np.sum(particles[0])/self.n_particle
        for t in np.arange(1,self.length):
            print("\r calculating... t={}".format(t), end="")
            #一期先予測
            predicts.append(self.one_predict(particles,t-1))
            #フィルタリング
            filtered=self.filtering(predicts,t-1)
            filtered_value[t]=filtered['filtering_value']
            resumple=self.resumpling(predicts[t-1],filtered['beta'])
            particles.append(resumple)
        return({'particles':particles,'predicts':predicts,'filtered_value':filtered_value})


# In[ ]:

model=ParticleFilter(data,10000,(2**-2)*(10**-1),2**-2)


# In[ ]:

result=model.simulate()


# In[ ]:

plt.figure(figsize=(20,9))
for i in range(len(pddata)):
    if i==0:
        plt.scatter(np.zeros(len(result['particles'][i]))+i,result['particles'][i],s=1,color='red',alpha=0.1,label='particle')
    plt.scatter(np.zeros(len(result['particles'][i]))+i,result['particles'][i],s=1,color='red',alpha=0.1)
plt.plot(pddata,color='blue',label='y')
plt.plot(result['filtered_value'],color='green',label='estimate')
plt.legend()
plt.title('particles = {}, upsilon2 = {}, omega2 = {}'.format(model.n_particle,model.upsilon2,model.omega2))
plt.show()

