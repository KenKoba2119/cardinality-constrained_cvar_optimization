import data
import model
import os
import numpy as np

def construct_string(n,seed,delta,alpha,data,k,rho):
    s = "Data." +data
    s = s + "_NumScenario." +str(n)
    s = s + "_alpha." +str(alpha)
    s = s + "_Delta." +str(delta)
    s = s + "_rho." +str(rho)
    s = s + "_K." +str(k)
    s = s + "_seed." +str(seed)
    return(s)
def create_directory(dir_name):
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    else:
        pass

def main(dataname,num_scenario,k,alpha,delta,rho,seed):
    DataName=dataname#port1"
    NumScenario = num_scenario#10000
    Seed = seed
    Delta = delta
    K = k
    Alpha = alpha#10
    timelimit = 3600
    ResultsDir = "../results/"

    create_directory(dir_name=ResultsDir)

    s = construct_string(n=NumScenario,
                         seed=Seed,
                         delta=Delta,
                         alpha=Alpha,
                         data=DataName,
                         k=K,rho=rho)
    s = ResultsDir+s

    if DataName in ["port1","port2","port5"]:
        Data = data.DataORLibrary("../data/orlibrary/"+DataName+".txt")
        mu = Data.get_mean_array()
        var = Data.get_variance_array()
        cov = Data.get_covariance_matrix()
    elif DataName in ["sp200"]:
        Data = data.SandP("../data/sandp/"+DataName+".csv")
        mu = Data.mean()
        cov = Data.cov()
    else:
        Data = data.FamaFrench("../data/famafrench/"+DataName+".csv")
        mu = Data.mean()
        cov = Data.cov()

    l,v = np.linalg.eig((cov+cov.T)/2)
    print("mu",mu)
    gamma = Alpha/np.sqrt(mu.shape[0])
    mu_scenario = data.generate_scenario(seed=Seed,mu=mu,cov=cov,size=NumScenario)
    p_list = [1/NumScenario for s in range(NumScenario)]

    liftbigm = model.LiftBigM(mu_scenario=mu_scenario,
                                             delta=Delta,p=p_list,k=K,gamma=gamma,timelimit=timelimit,
                                             mu=mu,rho=rho)
    liftsocp = model.LiftSocp(mu_scenario=mu_scenario,
                                                delta=Delta,p=p_list,k=K,gamma=gamma,timelimit=timelimit,
                                                mu=mu,rho=rho)
    cpa = model.CuttingPlaneAlgorithm(mu_scenario=mu_scenario,
                                                delta=Delta,p=p_list,k=K,gamma=gamma,timelimit=timelimit,
                                                mu=mu,rho=rho)
    bcpc = model.BilevelCuttingPlaneAlgorithmCallback(mu_scenario=mu_scenario,
                                               delta=Delta,
                                               p=p_list,
                                               gamma=gamma,k=K,timelimit=timelimit,
                                               mu=mu,rho=rho)
    bcp = model.BilevelCuttingPlaneAlgorithm(mu_scenario=mu_scenario,
                                               delta=Delta,
                                               p=p_list,
                                               gamma=gamma,k=K,timelimit=timelimit,
                                               mu=mu,rho=rho)

    cpbigm = model.CuttingPlaneBigM(mu_scenario=mu_scenario,
                                                    delta=Delta,
                                                    p=p_list,
                                                    gamma=gamma,k=K,timelimit=timelimit,
                                                    mu=mu,rho=rho)
    cpsocp = model.CuttingPlaneSocp(mu_scenario=mu_scenario,
                                    delta=Delta,
                                    p=p_list,
                                    gamma=gamma,k=K,timelimit=timelimit,
                                    mu=mu,rho=rho)
    bcpc.solve(logfilename=s+"_method.bcpc.log")
    bcpc.output_results(fname=s+"_method.bcpc.yml")#,z_init=z_init)

    bcp.solve(logfilename=s+"_method.bcp.log")
    bcp.output_results(fname=s+"_method.bcp.yml")#,z_init=z_init)

    cpbigm.solve(logfilename=s+"_method.cpbigm.log")
    cpbigm.output_results(fname=s+"_method.cpbigm.yml")#,z_init=z_init)

    cpsocp.solve(logfilename=s+"_method.cpsocp.log")#,z_init=z_init)
    cpsocp.output_results(fname=s+"_method.cpsocp.yml")#,z_init=z_init)

    liftbigm.solve(logfilename=s+"_method.liftbigm.log")#,z_init=z_init)
    liftbigm.output_results(fname=s+"_method.liftbigm.yml")#,z_init=z_init)

    liftsocp.solve(logfilename=s+"_method.liftsocp.log")#,z_init=z_init)
    liftsocp.output_results(fname=s+"_method.liftsocp.yml")#,z_init=z_init)

    cpa.solve(logfilename=s+"_method.cpa.log")#,z_init=z_init)
    cpa.output_results(fname=s+"_method.cpa.yml")#,z_init=z_init)


if __name__ == '__main__':
    d = "port1"#,"industry49"]#"port1"]#,"industry38"]
    main(dataname=d,num_scenario=10000,k=10,alpha=1,delta=0.9,rho=0.75,seed=0)
