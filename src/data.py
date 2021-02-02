import numpy as np
import pandas as pd

def generate_scenario(seed,mu,cov,size):
    np.random.seed(seed)
    return(np.random.multivariate_normal(mean=mu,cov=cov,size=size))

def choose_head_n_stock_data(fname,n):
    f = open(fname+".txt","r")
    wf = open(fname+"_nstock."+str(n) + ".txt","w")
    lines = f.readlines()
    i = 0
    wf.write(" "+str(n)+"\n")
    for line in lines:
        i += 1
        if i >= 2 and i <= n+1:
            wf.write(line)
        else:
            l = line.split(" ")[1:]
            if len(l) == 3:
                row = int(l[0])
                col = int(l[1])
                if row <= n and col <= n:
                    wf.write(line)
    f.close()
    wf.close()




class DataORLibrary():
    """docstring for ."""
    def __init__(self, fname):
        scale = 100
        self.fname = fname
        f = open(self.fname,"r")
        lines = f.readlines()
        lines =  [l.replace("\n","") for l in lines]
        lines =  [l[1:] for l in lines]

        self.n = int(lines[0])
        self.mean_list = []
        self.var_list = []
        self.cov_dict = {}
        for l in lines[:-1]:
            l = l.split(" ")
            if len(l) == 2:
                # mean and variance
                self.mean_list.append(float(l[0])*scale)
                self.var_list.append(float(l[1])*scale)
            if len(l) == 3:
                row = int(l[0])-1
                col = int(l[1])-1
                self.cov_dict[row,col] = float(l[2])*self.var_list[row]*self.var_list[col]
    def get_mean_array(self):
        return(np.array(self.mean_list))
    def get_variance_array(self):
        return(np.array(self.var_list))
    def get_covariance_matrix(self):
        m = np.empty((self.n,self.n))
        for key in self.cov_dict:
            row = key[0]
            col = key[1]
            val = self.cov_dict[row,col]
            m[row,col] = val
            m[col,row] = val
        l,v = np.linalg.eig((m+m.T)/2)
        print("aaaaaaaa")
        print("shape",m.shape)
        print("dict",self.cov_dict[self.n-1,self.n-1])
        print("n",self.n)
        print("lmin",np.min(l))
        return((m+m.T)/2)

class FamaFrench():
    """docstring for ."""
    def __init__(self,filename):
        if filename:
            self.read(filename)
        else:
            self.values = np.array([])
    def read(self,filename):
        '''read the dataset from a file'''
        self.dataframe = pd.read_csv(filename)
        self.dataframe['YYYYMM'] = self.dataframe['YYYYMM'].astype(int)
        self.yyyymm = self.dataframe.values[:,0]
        self.values = self.dataframe.values[:,1:].astype(float)
        #self.days =  self.dataframe.values[:,0]
        #self.days = self.days.astype(np.int64)
        print(self.values[1:3,:])
    def mean(self):
        return(np.mean(self.values,axis=0))
    def cov(self):
        return(np.cov(self.values.T))

class SandP():
    """docstring for ."""
    def __init__(self,filename):
        if filename:
            self.read(filename)
        else:
            self.values = np.array([])
    def read(self,filename):
        '''read the dataset from a file'''
        self.dataframe = pd.read_csv(filename)
        self.yyyymmdd = self.dataframe.values[:,0]
        self.values = self.dataframe.values[:,1:].astype(float)
        #print(self.values[1:3,1:3].shape)
    def mean(self):
        #print(self.values.shape)
        return(np.mean(self.values,axis=0))
    def cov(self):
        print(self.values.T.shape)
        #print("aaa")
        print(np.cov(self.values.T))
        #print("bbb")
        return(np.cov(self.values.T))

if __name__ == '__main__':
    #choose_head_n_stock_data(fname="../data/port2",n=40)
    #choose_head_n_stock_data(fname="../data/port2",n=50)
    #ort2 = DataORLibrary("../data/port2.txt")
    #cov = port2.get_covariance_matrix()
    #port2_40 = DataORLibrary("../data/port1.txt")
    #mu = port2_40.get_mean_array()
    #cov = port2_40.get_covariance_matrix()
    #print(np.max(mu),np.min(mu))
    #print(np.max(cov))
    #print(mu[0:15])
    #print(cov[0:15,0:15])
    #print(cov_40.shape)

    #print(cov[39,39],cov_40[39,39])
    d = SandP("../sandp/S&P200_return.csv")
    print(d.mean().shape,d.cov().shape)
    #print(np.max(d.mean()),np.min(d.mean()))
    #print(np.max(d.cov()))
