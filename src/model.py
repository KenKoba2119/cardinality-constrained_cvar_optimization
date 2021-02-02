import numpy as np
import gurobipy as grb
import time
import yaml
import osqp
import scipy as sp
from scipy import sparse

from itertools import chain,combinations
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
def get_unique_list(seq):
    seen = []
    return [x for x in seq if x not in seen and not seen.append(x)]
def construct_feasible_z(x,k):
    z = {i:0.0 for i in x}
    x_sorted = sorted(x.items(),key=lambda x:x[1],reverse=True)
    for key,val in x_sorted[0:k]:
        z[key]  = 1.0
    return(z)
def calc_minumu_return(mu,rho,k):
    mu_sorted = sorted(mu)
    mu_min = np.mean(mu_sorted[0:k])
    mu_max = np.mean(mu_sorted[-k:])
    return(mu_min+rho*(mu_max-mu_min))

class LiftBigM:
    def __init__(self,mu_scenario,delta,p,k,gamma,outputflag=1,timelimit=7200,mu=None,rho=None):
        self.mu_scenario = mu_scenario
        self.delta = delta
        self.p = p
        self.k = k
        self.gamma = gamma
        self.timelimit = timelimit
        self.num_stock = self.mu_scenario.shape[1]
        self.num_scenario = self.mu_scenario.shape[0]
        self.outputflag = outputflag
        self.mu_bar = None
        self.mu = mu
        self.rho = rho
        if self.rho != None:
            self.mu_bar = calc_minumu_return(mu=mu,rho=rho,k=self.k)
    def solve(self,logfilename):
        start_all = time.time()
        start_modeling = time.time()
        self.model = grb.Model("PrimalCVaRMinimization")
        x = {}
        q = {}
        z = {}
        for i in range(self.num_stock):
            x[i] = self.model.addVar(vtype="C",name="x_"+str(i),lb=0,ub=1)
            z[i] = self.model.addVar(vtype="B",name="z_"+str(i))
        for s in range(self.num_scenario):
            q[s] = self.model.addVar(vtype="C",name="q_"+str(s),lb=0)
        y = self.model.addVar(vtype="C",name="y",lb=-grb.GRB.INFINITY)
        self.model.update()
        self.model.addConstr(grb.quicksum(x[i] for i in range(self.num_stock))==1)
        self.model.addConstr(grb.quicksum(z[i] for i in range(self.num_stock))==self.k)
        if self.mu_bar != None:
            self.model.addConstr(grb.quicksum(self.mu[i]*x[i] for i in range(self.num_stock))>=self.mu_bar)
        for i in range(self.num_stock):
            self.model.addConstr(x[i] <= z[i])
        for s in range(self.num_scenario):
            self.model.addConstr(q[s] >= -grb.quicksum(self.mu_scenario[s,i]*x[i]
                                                    for i in range(self.num_stock))
                                    -y)
        self.model.update()
        self.model.setObjective(y+grb.quicksum(self.p[s]*q[s] for s in range(self.num_scenario))/(1-self.delta)+grb.quicksum(x[i]*x[i] for i in range(self.num_stock))/(2*self.gamma))
        self.model.params.OutputFlag = self.outputflag
        if self.outputflag == 1:
            self.model.params.LogFile = logfilename
        self.model.Params.TimeLimit=self.timelimit
        self.time_modeling = time.time()- start_modeling
        start_solve = time.time()
        self.model.optimize()
        self.time_solve = time.time()- start_solve
        self.time_all = time.time()- start_all
    def get_optimal_sol(self):
        if self.model.Status == grb.GRB.OPTIMAL:
            x_opt = {}
            z_opt = {}
            for var in self.model.getVars():
                if 'x_' in var.VarName:
                    i = int(var.VarName.split('_')[-1])
                    x_opt[i] = var.X
                if 'z_' in var.VarName:
                    i = int(var.VarName.split('_')[-1])
                    z_opt[i] = var.X
            return(x_opt,z_opt)
        else:
            return(None)
    def get_optimal_val(self):
        if self.model.Status == grb.GRB.OPTIMAL:
            return(self.model.objVal)
        else:
            return(None)
    def output_results(self,fname):
        d = {}
        d["params"] = {}
        d["params"]["gamma"] = float(self.gamma)
        d["params"]["alpha"] = float(self.gamma*np.sqrt(self.num_stock))
        d["params"]["num_scenario"] = int(self.num_scenario)
        d["params"]["num_stock"] = int(self.num_stock)
        d["params"]["delta"] = float(self.delta)
        d["params"]["k"] = int(self.k)
        d["params"]["rho"] = float(self.rho)
        d["results"] = {}
        if self.model.Status == grb.GRB.OPTIMAL:
            x_opt,z_opt = self.get_optimal_sol()
            d["results"]["status"] = "optimal"
            d["results"]["x_opt"] = {}
            for i in x_opt:
                d["results"]["x_opt"][i] = float(x_opt[i]*z_opt[i])
            d["results"]["opt_val"] = self.get_optimal_val()
        elif self.model.Status == grb.GRB.TIME_LIMIT:
            d["results"]["status"] = "timelimit"
            d["results"]["obj_bst"] = self.model.ObjVal
            d["results"]["obj_bnd"] = self.model.ObjBound
            d["results"]["gap"] = self.model.MIPGap
        d["results"]["time_all"] = self.time_all
        d["results"]["time_solve"] = self.time_solve
        with open(fname,"w") as f:
            yaml.dump(d,f,default_flow_style=False)
class LiftSocp:
    def __init__(self,mu_scenario,delta,p,k,gamma,timelimit=7200,mu=None,rho=None):
        self.mu_scenario = mu_scenario
        self.delta = delta
        self.p = p
        self.k = k
        self.gamma = gamma
        self.timelimit = timelimit
        self.num_stock = self.mu_scenario.shape[1]
        self.num_scenario = self.mu_scenario.shape[0]
        self.mu_bar = None
        self.mu = mu
        self.rho = rho
        if self.rho != None:
            self.mu_bar = calc_minumu_return(mu=mu,rho=rho,k=self.k)
    def solve(self,logfilename):
        start_all = time.time()
        start_modeling = time.time()
        self.model = grb.Model("PrimalSocpCVaRMinimization")
        x = {}
        theta = {}
        q = {}
        z = {}
        for i in range(self.num_stock):
            x[i] = self.model.addVar(vtype="C",name="x_"+str(i),lb=0,ub=1)
            theta[i] = self.model.addVar(vtype="C",name="theta_"+str(i),lb=0)
            z[i] = self.model.addVar(vtype="B",name="z_"+str(i))
        for s in range(self.num_scenario):
            q[s] = self.model.addVar(vtype="C",name="q_"+str(s),lb=0)
        y = self.model.addVar(vtype="C",name="y",lb=-grb.GRB.INFINITY)
        self.model.update()
        self.model.addConstr(grb.quicksum(x[i] for i in range(self.num_stock))==1)
        self.model.addConstr(grb.quicksum(z[i] for i in range(self.num_stock))==self.k)
        if self.mu_bar != None:
            self.model.addConstr(grb.quicksum(self.mu[i]*x[i] for i in range(self.num_stock))>=self.mu_bar)
        for i in range(self.num_stock):
            self.model.addConstr(x[i]*x[i] <= z[i]*theta[i])
        for s in range(self.num_scenario):
            self.model.addConstr(q[s] >= -grb.quicksum(self.mu_scenario[s,i]*x[i]
                                                    for i in range(self.num_stock))
                                    -y)
        self.model.update()
        self.model.setObjective(y+grb.quicksum(self.p[s]*q[s] for s in range(self.num_scenario))/(1-self.delta)+grb.quicksum(theta[i] for i in range(self.num_stock))/(2*self.gamma))
        self.model.Params.OutputFlag=1
        self.model.params.LogFile = logfilename
        self.model.Params.TimeLimit=self.timelimit
        self.model.update()
        self.time_modeling = time.time()-start_modeling
        self.model.optimize()
        self.time_all = time.time()-start_all
        self.time_solve = self.time_all - self.time_modeling
    def get_optimal_sol(self):
        if self.model.Status == grb.GRB.OPTIMAL:
            x_opt = {}
            z_opt = {}
            for var in self.model.getVars():
                if 'x_' in var.VarName:
                    i = int(var.VarName.split('_')[-1])
                    x_opt[i] = var.X
                if 'z_' in var.VarName:
                    i = int(var.VarName.split('_')[-1])
                    z_opt[i] = var.X
            return(x_opt,z_opt)
        else:
            return(None)
    def get_optimal_val(self):
        if self.model.Status == grb.GRB.OPTIMAL:
            return(self.model.objVal)
        else:
            return(None)
    def output_results(self,fname):
        d = {}
        d["params"] = {}
        d["params"]["gamma"] = float(self.gamma)
        d["params"]["alpha"] = float(self.gamma*np.sqrt(self.num_stock))
        d["params"]["num_scenario"] = int(self.num_scenario)
        d["params"]["num_stock"] = int(self.num_stock)
        d["params"]["delta"] = float(self.delta)
        d["params"]["rho"] = float(self.rho)
        d["params"]["k"] = int(self.k)
        d["results"] = {}
        if self.model.Status == grb.GRB.OPTIMAL:
            x_opt,z_opt = self.get_optimal_sol()
            d["results"]["status"] = "optimal"
            d["results"]["x_opt"] = {}
            for i in x_opt:
                d["results"]["x_opt"][i] = float(x_opt[i]*z_opt[i])
            d["results"]["opt_val"] = self.get_optimal_val()
        elif self.model.Status == grb.GRB.TIME_LIMIT:
            d["results"]["status"] = "timelimit"
            d["results"]["obj_bst"] = self.model.ObjVal
            d["results"]["obj_bnd"] = self.model.ObjBound
            d["results"]["gap"] = self.model.MIPGap
        d["results"]["time_all"] = self.time_all
        d["results"]["time_solve"] = self.time_solve
        d["results"]["time_modeling"] = self.time_modeling
        with open(fname,"w") as f:
            yaml.dump(d,f,default_flow_style=False)

class DualCVaRMinimizatin:
    def __init__(self,mu_scenario,delta,p,gamma,regularizer):
        self.mu_scenario = mu_scenario
        self.delta = delta
        self.p = p
        self.gamma = gamma
        self.regularizer = regularizer
    def solve(self):
        num_stock = self.mu_scenario.shape[1]
        num_scenario = self.mu_scenario.shape[0]
        self.model = grb.Model("DualCVaRMinimization")
        eta = {}
        xi = {}
        pi = {}
        w = {}
        for i in range(num_stock):
            w[i] = self.model.addVar(vtype="C",name="w_"+str(i),lb=-grb.GRB.INFINITY)
            #pi[i] = self.model.addVar(vtype="C",name="pi_"+str(i),lb=0)
        for s in range(num_scenario):
            eta[s] = self.model.addVar(vtype="C",name="eta_"+str(s),lb=0)
            #xi[s] = self.model.addVar(vtype="C",name="xi_"+str(s),lb=0)
        lambda_ = self.model.addVar(vtype="C",name="lambda",lb=-grb.GRB.INFINITY)
        self.model.update()
        self.model.addConstr(grb.quicksum(eta[s] for s in range(num_scenario))==1)
        for s in range(num_scenario):
            self.model.addConstr(eta[s]<=self.p[s]/(1-self.delta))
        if self.regularizer==False:
            for i in range(num_stock):
                self.model.addConstr(-grb.quicksum(eta[s]*self.mu_scenario[s,i] for s in range(num_scenario))-lambda_>=0)
            self.model.update()
            self.model.setObjective(lambda_, sense = grb.GRB.MAXIMIZE)
        if self.regularizer==True:
            for i in range(num_stock):
                self.model.addConstr(w[i]-grb.quicksum(eta[s]*self.mu_scenario[s,i] for s in range(num_scenario))-lambda_>=0)
                self.model.setObjective(-(self.gamma/2)*grb.quicksum(w[i]*w[i] for i in range(num_stock))+lambda_, sense = grb.GRB.MAXIMIZE)
        self.model.optimize()
    def solve_cp_representation(self):
        num_stock = self.mu_scenario.shape[1]
        num_scenario = self.mu_scenario.shape[0]
        J_list = list(powerset([i for i in range(num_scenario)]))
        self.model = grb.Model("DualCVaRMinimization")
        zeta = {}
        w = {}
        for i in range(num_stock):
            w[i] = self.model.addVar(vtype="C",name="w_"+str(i),lb=-grb.GRB.INFINITY)
        for J in J_list:
            zeta[J] = self.model.addVar(vtype="C",name="zeta_"+str(J),lb=0)
        lambda_ = self.model.addVar(vtype="C",name="lambda",lb=-grb.GRB.INFINITY)
        self.model.update()
        self.model.addConstr(grb.quicksum(zeta[J] for J in J_list)<=1/(1-self.delta))
        self.model.addConstr(1 - grb.quicksum(zeta[J]*sum([self.p[s] for s in J]) for J in J_list)==0)
        for i in range(num_stock):
            self.model.addConstr(w[i] >= grb.quicksum(zeta[J]*grb.quicksum(self.p[s]*self.mu_scenario[s,i] for s in J) for J in J_list)+lambda_)
        self.model.setObjective(-(self.gamma/2)*grb.quicksum(w[i]*w[i] for i in range(num_stock))+lambda_, sense = grb.GRB.MAXIMIZE)
        self.model.optimize()
    def get_optimal_sol(self):
        if self.model.Status == grb.GRB.OPTIMAL:
            self.w_opt = {}
            self.x_opt = {}
            for var in self.model.getVars():
                if 'w_' in var.VarName:
                    i = int(var.VarName.split('_')[-1])
                    self.w_opt[i] = var.X
                    self.x_opt[i] = self.w_opt[i] * self.gamma
            return(self.x_opt)
        else:
            return(None)

class CuttingPlaneAlgorithm:
    def __init__(self,mu_scenario,delta,k,p,gamma,timelimit=7200,mu=None,rho=None):
        self.mu_scenario = mu_scenario
        self.delta = delta
        self.p = p
        self.k = k # max cardinality
        self.gamma = gamma # parameter of regularizer
        self.num_stock = self.mu_scenario.shape[1]
        self.num_scenario = self.mu_scenario.shape[0]
        self.time_solve_inner_prob = 0
        self.timelimit = timelimit
        self.mu_bar = None
        self.mu = mu
        self.rho = rho
        if self.rho != None:
            self.mu_bar = calc_minumu_return(mu=mu,rho=rho,k=self.k)
    def solve_inner_prob(self,z):
        start_solve_inner_prob = time.time()
        selected_index_set = set([i for i in z if z[i]>= 0.5 ])
        model_inner = grb.Model("inner_prob")
        eta = {}
        w = {}
        for i in selected_index_set:
            w[i] = model_inner.addVar(vtype="C",name="w_"+str(i),lb=-grb.GRB.INFINITY)
        for s in range(self.num_scenario):
            eta[s] = model_inner.addVar(vtype="C",name="eta_"+str(s),lb=0,ub=self.p[s]/(1-self.delta))
        lambda_ = model_inner.addVar(vtype="C",name="lambda",lb=-grb.GRB.INFINITY)
        model_inner.update()
        model_inner.addConstr(grb.quicksum(eta[s] for s in range(self.num_scenario))==1)
        if self.mu_bar == None:
            for i in selected_index_set:
                model_inner.addConstr(w[i]-grb.quicksum(eta[s]*self.mu_scenario[s,i] for s in range(self.num_scenario))-lambda_>=0)
            model_inner.setObjective(-(self.gamma/2)*grb.quicksum(w[i]*w[i] for i in selected_index_set)+lambda_, sense = grb.GRB.MAXIMIZE)
        else:
            beta_l = model_inner.addVar(vtype="C",name="beta",lb=0)
            for i in selected_index_set:
                model_inner.addConstr(w[i]>=grb.quicksum(eta[s]*self.mu_scenario[s,i] for s in range(self.num_scenario))+self.mu[i]*beta_l+lambda_)
            model_inner.setObjective(-(self.gamma/2)*grb.quicksum(w[i]*w[i] for i in selected_index_set)+self.mu_bar*beta_l+lambda_, sense = grb.GRB.MAXIMIZE)
        model_inner.Params.OutputFlag=0
        model_inner.Params.NumericFocus=0
        model_inner.update()
        model_inner.optimize()
        if model_inner.Status == grb.GRB.OPTIMAL:
            opt_val = model_inner.objVal
            w_opt = {}
            eta_opt = {}
            lambda_opt = {}
            for var in model_inner.getVars():
                if "w_" in var.VarName:
                    i = int(var.VarName.split('_')[-1])
                    w_opt[i] = var.X
                if "eta_" in var.VarName:
                    s = int(var.VarName.split('_')[-1])
                    eta_opt[s] = var.X
                if "lambda" in var.VarName:
                    lambda_opt = var.X
                if "beta" in var.VarName:
                    beta_l_opt = var.X
            for i in range(self.num_stock):
                if i not in w_opt:
                    if self.mu_bar == None:
                        tmp = sum([eta_opt[s]*self.mu_scenario[s,i] for s in range(self.num_scenario)])
                    else:
                        tmp = sum([eta_opt[s]*self.mu_scenario[s,i] for s in range(self.num_scenario)])+self.mu[i]*beta_l_opt
                    tmp = tmp+lambda_opt
                    w_opt[i] = max(0,tmp)
        else:
            w_opt = None
            opt_val = None
        self.time_solve_inner_prob = self.time_solve_inner_prob  +(time.time() - start_solve_inner_prob)
        return(w_opt,opt_val)
    def solve(self,logfilename):
        start_all = time.time()
        start_prep = time.time()
        socp_conti =LiftBigM(mu_scenario=self.mu_scenario,
                                              delta=self.delta,
                                              p=self.p,
                                              k=self.num_stock,
                                              gamma=self.gamma,outputflag=0)
        socp_conti.solve(logfilename=None)
        theta_lower = socp_conti.get_optimal_val()
        self.time_prep = time.time() - start_prep
        self.time_callback = 0
        self.num_callback = 0
        def add_cutting_plane(model,where):
            if where  == grb.GRB.callback.MIPSOL:
                start_callback = time.time()
                z_tmp = {}
                z_var = {}
                for var in model._vars:
                    if "z" in var.VarName:
                        i = int(var.VarName.split("_")[-1])
                        z_tmp[i] = model.cbGetSolution(var)
                        z_var[i] = var
                    if "theta" in var.VarName:
                        theta_var = var
                w_tmp,f = self.solve_inner_prob(z=z_tmp)
                if w_tmp != None:
                    model.cbLazy(theta_var >= f + grb.quicksum(-(self.gamma/2)*(w_tmp[i]*w_tmp[i])*(z_var[i]-z_tmp[i]) for i in range(self.num_stock)))
                else:
                    model.cbLazy(grb.quicksum(z_tmp[i]*(1-z_var[i])
                          + (1-z_tmp[i])*z_var[i] for i in range(self.num_stock))>=1)
                self.time_callback += time.time() - start_callback
                self.num_callback += 1
        start_modeling = time.time()
        self.model = grb.Model("CuttingPlaneCVaRMinimization")
        z = {}
        for i in range(self.num_stock):
            z[i] = self.model.addVar(vtype="B",name="z_"+str(i))
        theta = self.model.addVar(vtype="C",name="theta",lb=theta_lower)
        self.model.update()
        self.model.addConstr(grb.quicksum(z[i] for i in range(self.num_stock))==self.k)
        self.model.update()
        self.model.setObjective(theta)
        self.model._vars = self.model.getVars()
        self.model.params.NumericFocus = 1
        self.model.params.LazyConstraints = 1
        self.model.params.NumericFocus = 3
        self.model.params.LogFile = logfilename
        self.model.params.TIME_LIMIT = self.timelimit
        self.model.update()
        self.time_modeling = time.time()-start_modeling
        start_solve = time.time()
        self.model.optimize(add_cutting_plane)
        self.time_solve = time.time()-start_solve
        self.time_all = time.time() - start_all
    def get_optimal_sol(self):
        if self.model.Status == grb.GRB.OPTIMAL:
            x_opt = {}
            z_opt = {}
            w_opt = {}
            for var in self.model.getVars():
                if 'z_' in var.VarName:
                    i = int(var.VarName.split('_')[-1])
                    z_opt[i] = var.X
            w_opt,f = self.solve_inner_prob(z=z_opt)
            for i in w_opt:
                x_opt[i] = self.gamma * z_opt[i] * w_opt[i]
            return(x_opt,z_opt)
        else:
            return(None,None)
    def get_optimal_val(self):
        if self.model.Status == grb.GRB.OPTIMAL:
            return(self.model.objVal)
        else:
            return(None)
    def output_results(self,fname):
        d = {}
        d["params"] = {}
        d["params"]["gamma"] = float(self.gamma)
        d["params"]["alpha"] = float(self.gamma*np.sqrt(self.num_stock))
        d["params"]["num_scenario"] = int(self.num_scenario)
        d["params"]["num_stock"] = int(self.num_stock)
        d["params"]["delta"] = float(self.delta)
        d["params"]["rho"] = float(self.rho)
        d["params"]["k"] = int(self.k)
        d["results"] = {}
        if self.model.Status == grb.GRB.OPTIMAL:
            x_opt, z_opt = self.get_optimal_sol()
            d["results"]["status"] = "optimal"
            d["results"]["x_opt"] = {}
            for i in x_opt:
                d["results"]["x_opt"][i] = float(x_opt[i]*z_opt[i])
            d["results"]["opt_val"] = self.get_optimal_val()
        elif self.model.Status == grb.GRB.TIME_LIMIT:
            d["results"]["status"] = "timelimit"
            d["results"]["status"] = "timelimit"
            d["results"]["obj_bst"] = self.model.ObjVal
            d["results"]["obj_bnd"] = self.model.ObjBound
            d["results"]["gap"] = self.model.MIPGap
        d["results"]["time_all"] = self.time_all
        d["results"]["time_prep"] = self.time_prep
        d["results"]["time_solve"] = self.time_solve
        d["results"]["time_modeling"] = self.time_modeling
        d["results"]["time_callback"] = self.time_callback
        d["results"]["num_callback"] = self.num_callback
        with open(fname,"w") as f:
            yaml.dump(d,f,default_flow_style=False)


class BilevelCuttingPlaneAlgorithm:
    def __init__(self,mu_scenario,delta,k,p,gamma,timelimit=7200,mu=None,rho=None):
        self.mu_scenario = mu_scenario
        self.delta = delta
        self.p = p
        self.k = k # max cardinality
        self.gamma = gamma # parameter of regularizer
        self.num_stock = self.mu_scenario.shape[1]
        self.num_scenario = self.mu_scenario.shape[0]
        self.weighted_mu_scenario = np.array([self.p[s]*self.mu_scenario[s,:] for s in range(self.num_scenario)])
        self.status = ""
        self.timelimit=timelimit
        self.objub_list = []
        self.r_tol = 10**(-5)*(1-self.delta)
        self.mu_bar = None
        self.mu = mu
        self.rho = rho
        if rho != None:
            self.mu_bar = calc_minumu_return(mu=self.mu,rho=self.rho,k=self.k)
    def solve(self,logfilename,z_init=None):
        start_all = time.time()
        start_prep = time.time()
        dcp = BilevelCuttingPlaneAlgorithmCallback(mu_scenario=self.mu_scenario,
                                                   delta=self.delta,
                                                   p=self.p,
                                                   k=self.k,
                                                   gamma=self.gamma,mu=self.mu,rho=self.rho)
        J_list_conti,x_conti,r_tmp,y_tmp,theta_lower,r_hat  = dcp.solve_inner_primal_problem(selected_index=[i for i in range(self.num_stock)],
                                                                    J_list_init=[[s for s in range(self.num_scenario)]],
                                                                    max_loop=10000,tol=self.r_tol)
        w_conti,theta_conti = dcp.solve_inner_dual_problem(J_list = J_list_conti,#self.J_pos,
                                                      selected_index=[i for i in range(self.num_stock)])
        z_tmp = construct_feasible_z(x=x_conti,k=self.k)
        selected_index = [i for i in z_tmp if z_tmp[i]>= 0.5]
        J_list,x_tmp,r_tmp,y_tmp,theta_upper,r_hat = dcp.solve_inner_primal_problem(selected_index=selected_index,
                                        J_list_init=[[s for s in range(self.num_scenario)]],
                                        max_loop=10000,tol=self.r_tol)
        w_feas,f = dcp.solve_inner_dual_problem(J_list = J_list,#self.J_pos,
                                                              selected_index=selected_index)
        theta_upper = dcp.calc_obj_ub(x=x_tmp)
        self.time_prep = time.time() - start_prep
        J_list = J_list_conti#[[s for s in range(self.num_scenario)]]
        #self.J_list = J_list
        self.f_list = []
        theta_list = []
        objbnd_list = []
        #self.J_pos = self.J_list
        start_modeling = time.time()
        master_model = grb.Model("CuttingPlaneCVaRMinimization")
        theta = master_model.addVar(vtype="C",name="theta",lb=theta_lower)
        z = {}
        for i in range(self.num_stock):
            z[i] = master_model.addVar(vtype="B",name="z_"+str(i))
        master_model.update()
        master_model.addConstr(grb.quicksum(z[i] for i in range(self.num_stock))==self.k)
        master_model.update()
        master_model.setObjective(theta)
        master_model.update()
        #master_model.addConstr(theta >= theta_conti)
        master_model.addConstr(theta >= f
            + grb.quicksum(-(self.gamma/2)*(w_feas[i]*w_feas[i])*(z[i]-z_tmp[i]) for i in range(self.num_stock)))
        #master_model.addConstr(theta >= theta_conti
        #    + grb.quicksum(-(self.gamma/2)*(w_conti[i]*w_conti[i])*(z[i]-x_conti[i]) for i in range(self.num_stock)))
        self.time_modeling = time.time() - start_modeling
        start_solve = time.time()
        master_model.params.LogFile = logfilename
        pre_gap = 100
        master_model.setObjective(theta)
        for i in range(self.num_stock):
            z[i].start = z_tmp[i]
            master_model.update()
        for loop in range(10000):
            master_model.params.TIME_LIMIT = max([self.timelimit - (time.time()-start_all),1])
            master_model.update()
            master_model.Params.OutputFlag=0
            master_model.optimize()
            if master_model.Status == grb.GRB.OPTIMAL:
                z_pos = {i: z_tmp[i] for i in z_tmp}
                z_tmp = {}
                for var in master_model.getVars():
                    if 'z_' in var.VarName:
                        i = int(var.VarName.split('_')[-1])
                        z_tmp[i] = var.X
                    if 'theta' in var.VarName:
                        theta_tmp = var.X
                selected_index = [i for i in z_tmp if z_tmp[i]>= 0.5]
                J_list,x_tmp,r_tmp,y_tmp,o,r_hat =dcp.solve_inner_primal_problem(selected_index=selected_index,
                                                J_list_init=[[s for s in range(self.num_scenario)]],
                                                max_loop=10000,tol=self.r_tol)
                if J_list != None:
                    w_tmp,f = dcp.solve_inner_dual_problem(J_list = J_list,#self.J_pos,
                                                            selected_index=selected_index)
                    ub = dcp.calc_obj_ub(x=x_tmp)
                    self.f_list.append(f)
                    self.objub_list.append(ub)
                    self.theta_tmp = theta_tmp
                    gap = np.abs(theta_tmp-np.min(self.objub_list))/(np.min(self.objub_list)+1)

                    if np.abs(gap) <= 10**(-5) or int(sum(abs(z_tmp[i]-z_pos[i]) for i in z_tmp))==0:
                        self.status = "optimal"
                        self.x_opt = {i: x_tmp[i] for i in x_tmp}
                        self.z_opt = {i: z_tmp[i] for i in z_tmp}
                        self.theta_opt = theta_tmp
                        break
                    else:
                        theta.start = f
                        for i in range(self.num_stock):
                            z[i].start = z_tmp[i]
                        master_model.addConstr(theta >= f + grb.quicksum(-(self.gamma/2)*(w_tmp[i]*w_tmp[i])*(z[i]-z_tmp[i]) for i in range(self.num_stock)))
                    pre_gap = gap
                else:
                    master_model.addConstr(grb.quicksum(z_tmp[i]*(1-z[i])
                          + (1-z_tmp[i])*z[i] for i in range(self.num_stock))>=1)
                if time.time()-start_all > self.timelimit:
                    self.status = "timelimit"
                    break
            elif master_model.Status == grb.GRB.INFEASIBLE:
                master_model.addConstr(grb.quicksum(z_tmp[i]*(1-z[i])
                      + (1-z_tmp[i])*z[i] for i in range(self.num_stock))>=1)
            else:
                self.status = "timelimit"
                break
        self.time_solve = time.time() -start_solve
        self.time_all = time.time()-start_all
        self.loop = loop+1
    def output_results(self,fname):
        d = {}
        d["params"] = {}
        d["params"]["gamma"] = float(self.gamma)
        d["params"]["alpha"] = float(self.gamma*np.sqrt(self.num_stock))
        d["params"]["num_scenario"] = int(self.num_scenario)
        d["params"]["num_stock"] = int(self.num_stock)
        d["params"]["delta"] = float(self.delta)
        d["params"]["rho"] = float(self.rho)
        d["params"]["k"] = int(self.k)
        #d["params"]["tau"] = self.tau
        d["results"] = {}
        if self.status == "optimal":
            d["results"]["status"] = "optimal"
            d["results"]["x_opt"] = {i:self.x_opt[i] if i in self.x_opt else 0 for i in range(self.num_stock)}
            d["results"]["theta_opt"] = self.theta_opt
        elif self.status == "timelimit":
            d["results"]["status"] = "timelimit"
            d["results"]["theta_upper"] = float(np.min(self.objub_list))
            d["results"]["theta_lower"] = float(self.theta_tmp)
        d["results"]["time_all"] = self.time_all
        d["results"]["time_solve"] = self.time_solve
        d["results"]["time_prep"] = self.time_prep
        d["results"]["time_modeling"] = self.time_modeling
        d["results"]["num_loop"] = self.loop
        with open(fname,"w") as f:
            yaml.dump(d,f,default_flow_style=False)


class BilevelCuttingPlaneAlgorithmCallback:
    def __init__(self,mu_scenario,delta,k,p,gamma,timelimit=7200,mu=None,rho=None):
        self.mu_scenario = mu_scenario
        self.delta = delta
        self.p = p
        self.k = k # max cardinality
        self.gamma = gamma # parameter of regularizer
        self.num_stock = self.mu_scenario.shape[1]
        self.num_scenario = self.mu_scenario.shape[0]
        self.time_primal = 0
        self.time_dual = 0
        self.time_callback = 0
        self.num_callback = 0
        self.weighted_mu_scenario = np.array([self.p[s]*self.mu_scenario[s,:] for s in range(self.num_scenario)])
        self.timelimit = timelimit
        self.r_tol = 10**(-5)*(1-self.delta)
        self.mu_bar = None
        self.mu=mu
        self.rho=rho
        if rho != None:
            self.mu_bar = calc_minumu_return(mu=self.mu,rho=self.rho,k=self.k)
    def get_tmp_sol(self,model):
        if model.Status == grb.GRB.OPTIMAL:
            x_opt = {}
            for var in model.getVars():
                if 'x_' in var.VarName:
                    i = int(var.VarName.split('_')[-1])
                    x_opt[i] = var.X
                if 'r' in var.VarName:
                    r_opt = var.X
                if 'y' in var.VarName:
                    y_opt = var.X
            return(x_opt,r_opt,y_opt)
        else:
            return(None,None,None)
    def check_feasibility(self,x_tmp,r_tmp,y_tmp):
        S_star =[]
        a  = time.time()
        x_vec = np.array([x_tmp[i] if i in x_tmp else 0 for i in range(self.num_stock) ])
        v_vec = -np.dot(self.mu_scenario,x_vec)-y_tmp
        S_star = [s for s in range(self.num_scenario) if v_vec[s] > 0]
        r_hat = sum(self.p[s]*(-sum(self.mu_scenario[s,i]*x_tmp[i] for i in x_tmp)-y_tmp) for s in S_star)
        return(S_star,r_hat)

    def check_active_scenario(self,x,r,y,J_list,A_mat,b_vec):
        return([j for j in range(len(J_list)) if np.abs(r-sum([-A_mat[j,i]*x[i] for i in x])-b_vec[j]*y)>=10**(-1)])

    def calc_obj_ub(self,x):
        x_vec = np.array([x[i] if i in x else 0 for i in range(self.num_stock) ])
        loss_vec = -np.dot(self.mu_scenario,x_vec)
        var = np.percentile(loss_vec, self.delta*100)
        diff_vec_over_var = np.array([loss_vec[i]-var for i in range(self.num_scenario) if loss_vec[i] > var])
        cvar = var+np.mean(diff_vec_over_var)
        return(cvar+np.sum([x_vec[i]**2  for i in range(self.num_stock)])/(2*self.gamma))

    def solve_inner_primal_problem(self,selected_index,J_list_init,max_loop,tol=10**(-5)):
        start_primal = time.time()
        model = grb.Model("inner_prob_primal")
        x = {}
        for i in selected_index:
            x[i] =model.addVar(vtype="C",name="x_"+str(i),lb=0,ub=1)
        y = model.addVar(vtype="C",name="y",lb=-grb.GRB.INFINITY)
        r = model.addVar(vtype="C",name="r",lb=0)
        model.update()
        J_list = J_list_init
        A_mat = np.array([np.sum([self.weighted_mu_scenario[s,:]
                         for s in J_list[j]],axis=0) for j in range(len(J_list))])
        b_vec = np.array([np.sum([self.p[s] for s in J_list[j]]) for j in range(len(J_list))])
        for j in range(len(J_list)):
            model.addConstr(r>=grb.quicksum(-A_mat[j,i]*x[i] for i in x)-b_vec[j]*y,name="zeta_"+str(j))
        model.addConstr(grb.quicksum(x[i] for i in x)==1,name="lambda")
        if self.mu_bar != None:
            model.addConstr(grb.quicksum(self.mu[i]*x[i] for i in x) >= self.mu_bar,name="beta")
        model.setObjective(grb.quicksum(x[i]*x[i] for i in x)/(2*self.gamma)+y+r/(1-self.delta))
        model.Params.OutputFlag=0
        #model.Params.Method = 1
        model.Params.DualReductions=1
        #model.Params.NumericFocus=3
        model.update()
        # check feasibility
        if np.max([self.mu[i] for i in x]) >= self.mu_bar:
            for loop in range(max_loop):
                model.optimize()
                x_tmp ,r_tmp,y_tmp= self.get_tmp_sol(model)
                if x_tmp == None:
                    break
                S_star, r_hat = self.check_feasibility(x_tmp=x_tmp,
                                                       r_tmp =r_tmp,
                                                       y_tmp = y_tmp)
                if r_tmp-r_hat >= -tol:
                    break
                else:
                    J_list.append(S_star)
                    A_vec = np.sum([self.weighted_mu_scenario[s,:] for s in S_star],axis=0)
                    b = np.sum([self.p[s] for s in S_star])
                    model.addConstr(r>=grb.quicksum(-A_vec[i]*x[i] for i in x)-b*y,name="zeta_"+str(len(J_list)-1))
            self.time_primal += time.time()-start_primal
            if x_tmp == None:
                return(None,None,None,None,None,None)
            else:
                opt_val = model.objVal
                return(J_list,x_tmp,r_tmp,y_tmp,opt_val,r_hat)
        else:
            return(None,None,None,None,None,None)
    def solve_inner_dual_problem(self,J_list,selected_index):
        start_dual = time.time()
        model = grb.Model("DualCVaRMinimization")
        start_modeling = time.time()
        A_mat = np.array([np.sum([self.weighted_mu_scenario[s,:]
                         for s in J_list[j]],axis=0) for j in range(len(J_list))])
        b_vec = np.array([np.sum([self.p[s] for s in J_list[j]]) for j in range(len(J_list))])
        zeta = {}
        w = {}
        for i in selected_index:
            w[i] = model.addVar(vtype="C",name="w_"+str(i),lb=-grb.GRB.INFINITY)
        for j in range(len(J_list)):
            zeta[j] = model.addVar(vtype="C",name="zeta_"+str(j),lb=0)
        lambda_ = model.addVar(vtype="C",name="lambda",lb=-grb.GRB.INFINITY)
        model.update()
        model.addConstr(grb.quicksum(zeta[j] for j in range(len(J_list)))<=1/(1-self.delta)) #
        model.addConstr(1 - grb.quicksum(zeta[j]*b_vec[j] for j in range(len(J_list)))==0)
        if self.mu_bar == None:
            for i in w:
                model.addConstr(w[i] >= grb.quicksum(zeta[j]*A_mat[j,i] for j in range(len(J_list)))+lambda_)
            model.setObjective(-(self.gamma/2)*grb.quicksum(w[i]*w[i] for i in w)+lambda_, sense = grb.GRB.MAXIMIZE)
        elif self.mu_bar != None:
            beta_l = model.addVar(vtype="C",name="beta",lb=0)
            for i in w:
                model.addConstr(w[i] >= grb.quicksum(zeta[j]*A_mat[j,i] for j in range(len(J_list)))+self.mu[i]*beta_l+lambda_)
            model.setObjective(-(self.gamma/2)*grb.quicksum(w[i]*w[i] for i in w)+self.mu_bar*beta_l+lambda_,sense = grb.GRB.MAXIMIZE)
        model.Params.OutputFlag=0
        model.optimize()
        if model.Status == grb.GRB.OPTIMAL:
            opt_val = model.objVal
            w_opt = {}
            zeta_opt = {}
            for var in model.getVars():
                if "w_" in var.VarName:
                    i = int(var.VarName.split('_')[-1])
                    w_opt[i] = var.X
                if "zeta_" in var.VarName:
                    j = int(var.VarName.split('_')[-1])
                    zeta_opt[j] = var.X
                if "lambda" in var.VarName:
                    lambda_opt = var.X
                if "beta" in var.VarName:
                    beta_l_opt = var.X
            for i in range(self.num_stock):
                if i not in w_opt:
                    tmp = sum(zeta_opt[j]*A_mat[j,i] for j in range(len(J_list)))
                    if self.mu_bar == None:
                        tmp = tmp+lambda_opt
                    elif self.mu_bar != None:
                        tmp = tmp + self.mu[i] * beta_l_opt + lambda_opt
                    w_opt[i] = max(0,tmp)
            self.time_dual += time.time()-start_dual
            return(w_opt,opt_val)
        else:
            return(None,None)
    def remove_redundant_J(self,x_tmp,r_tmp,y_tmp,J_list):
        start = time.time()
        x_vec = np.array([x_tmp[i] if i in x_tmp else 0 for i in range(self.num_stock) ])
        r_list = [sum(self.p[s]*(-sum(self.mu_scenario[s,i]*x_tmp[i] for i in x_tmp)-y_tmp) for s in J_list[j]) for j in range(len(J_list))]
        j_list = [j for j in range(len(r_list)) if r_tmp - r_list[j] <= 10**(-3)]
        return([J_list[j] for j in j_list])

    def solve(self,logfilename,z_init=None):
        start_all = time.time()
        start_prep = time.time()
        J_list_conti,x_conti,r_tmp,y_tmp,theta_lower,r_hat = self.solve_inner_primal_problem(selected_index=[i for i in range(self.num_stock)],
                                                                    J_list_init=[[s for s in range(self.num_scenario)]],
                                                                    max_loop = 10000,tol=self.r_tol)
        w_conti,theta_conti = self.solve_inner_dual_problem(J_list = J_list_conti,#self.J_pos,
                                                            selected_index=[i for i in range(self.num_stock)])
        z_feas = construct_feasible_z(x=x_conti,k=self.k)
        selected_index = [i for i in z_feas if z_feas[i]>= 0.5]
        J_list,x_tmp,r_tmp,y_ymp,theta_upper,r_hat = self.solve_inner_primal_problem(selected_index=selected_index,
                                        J_list_init=J_list_conti,#[[s for s in range(self.num_scenario)]],
                                        max_loop = 10000,tol=self.r_tol)
        w_feas,theta_upper = self.solve_inner_dual_problem(J_list = J_list,#self.J_pos,
                                                           selected_index=selected_index)
        self.time_prep = time.time() - start_prep
        start_modeling = time.time()
        self.time_callback = 0
        self.num_callback = 0
        self.J_list = [[s for s in range(self.num_scenario)]]
        self.f_list = []
        self.theta_list = []
        self.objbnd_list = []
        self.obj_ub_list = []
        self.theta_list = []
        self.z_pos = []
        self.J_list = J_list_conti
        self.J_list_conti = J_list_conti
        self.theta_upper = theta_upper
        def add_cutting_plane(model,where):
            if where  == grb.GRB.callback.MIPSOL:
                self.num_callback += 1
                start_callback = time.time()
                z_tmp = {}
                z_var = {}
                for var in model._vars:
                    if "z" in var.VarName:
                        i = int(var.VarName.split("_")[-1])
                        z_tmp[i] = model.cbGetSolution(var)
                        z_var[i] = var
                    if "theta" in var.VarName:
                        theta_tmp = model.cbGetSolution(var)
                        theta_var = var
                selected_index = [i for i in z_tmp if z_tmp[i]>= 0.5]
                self.J_list,x_tmp,r_tmp,y_tmp,o,r_hat = self.solve_inner_primal_problem(selected_index=selected_index,
                                                J_list_init=[[s for s in range(self.num_scenario)]],
                                                max_loop=100000,tol=self.r_tol)#self.r_tol)#*(1+10/np.sqrt(self.num_callback)))#10**(-2))#
                # feasible z
                if self.J_list != None:
                    w_tmp,f = self.solve_inner_dual_problem(J_list = self.J_list,#self.J_pos,
                                                            selected_index=selected_index)
                    model.cbLazy(theta_var >= np.min([o,f]) + grb.quicksum(-(self.gamma/2)*(w_tmp[i]*w_tmp[i])*(z_var[i]-z_tmp[i]) for i in range(self.num_stock)))
                    theta_ub = self.calc_obj_ub(x=x_tmp)
                    self.obj_ub_list.append(theta_ub)


                    self.time_callback += time.time() - start_callback
                else:
                    model.cbLazy(grb.quicksum(z_tmp[i]*(1-z_var[i])
                          + (1-z_tmp[i])*z_var[i] for i in range(self.num_stock))>=1)

        # formulate master problem
        self.master_model = grb.Model("CuttingPlaneCVaRMinimization")
        theta = self.master_model.addVar(vtype="C",name="theta",lb=theta_lower,ub=2*theta_upper)
        z = {}
        for i in range(self.num_stock):
            z[i] = self.master_model.addVar(vtype="B",name="z_"+str(i))
        self.master_model.update()
        self.master_model.addConstr(grb.quicksum(z[i] for i in range(self.num_stock))==self.k)
        self.master_model.addConstr(theta >= theta_upper
            + grb.quicksum(-(self.gamma/2)*(w_feas[i]*w_feas[i])*(z[i]-z_feas[i]) for i in range(self.num_stock)))
        self.master_model.update()
        self.master_model.setObjective(theta)
        theta.start = theta_upper
        self.master_model.update()
        for i in range(self.num_stock):
            z[i].start = z_feas[i]
            self.master_model.update()
        self.master_model.update()
        self.master_model._vars = self.master_model.getVars()
        self.master_model.params.LazyConstraints = 1
        self.master_model.params.LogFile = logfilename
        self.master_model.params.TIME_LIMIT = self.timelimit
        self.master_model.update()
        self.time_modeling  =  time.time() - start_modeling
        start_solve = time.time()
        self.master_model.optimize(add_cutting_plane)
        self.time_solve = time.time()-start_solve
        self.time_all = time.time()-start_all
    def get_optimal_sol(self):
        if self.master_model.Status == grb.GRB.OPTIMAL:
            x_opt = {}
            z_opt = {}
            w_opt = {}
            for var in self.master_model.getVars():
                if 'z_' in var.VarName:
                    i = int(var.VarName.split('_')[-1])
                    z_opt[i] = var.X
            selected_index = [i for i in z_opt if z_opt[i]>= 0.5]
            J,x_opt,r_tmp,y_tmp,opt_val,r_hat = self.solve_inner_primal_problem(selected_index = selected_index,
                                                      J_list_init=[[s for s in range(self.num_scenario)]],
                                                      tol=10**(-6),max_loop=10000)
            for i in range(self.num_stock):
                if i in x_opt:
                    pass
                else:
                    x_opt[i] = 0
            return(x_opt,z_opt)
        else:
            return(None)

    def get_optimal_val(self):
        if self.master_model.Status == grb.GRB.OPTIMAL:
            return(self.master_model.objVal)
        else:
            return(None)

    def output_results(self,fname):
        d = {}
        d["params"] = {}
        d["params"]["gamma"] = float(self.gamma)
        d["params"]["alpha"] = float(self.gamma*np.sqrt(self.num_stock))
        d["params"]["num_scenario"] = int(self.num_scenario)
        d["params"]["num_stock"] = int(self.num_stock)
        d["params"]["delta"] = float(self.delta)
        d["params"]["rho"] = float(self.rho)
        d["params"]["k"] = int(self.k)
        d["results"] = {}
        if self.master_model.Status == grb.GRB.OPTIMAL:
            x_opt, z_opt = self.get_optimal_sol()
            d["results"]["status"] = "optimal"
            d["results"]["x_opt"] = {}
            for i in x_opt:
                d["results"]["x_opt"][i] = float(x_opt[i]*z_opt[i])
            d["results"]["opt_val"] = self.get_optimal_val()
        elif self.master_model.Status == grb.GRB.TIME_LIMIT:
            d["results"]["status"] = "timelimit"
            d["results"]["obj_bst"] = self.master_model.ObjVal
            d["results"]["obj_bnd"] = self.master_model.ObjBound
            d["results"]["gap"] = self.master_model.MIPGap
        d["results"]["obj_bst_manually_calculed"] = float(np.min(self.obj_ub_list))
        d["results"]["time_all"] = self.time_all
        d["results"]["time_prep"] = self.time_prep
        d["results"]["time_modeing"] = self.time_modeling
        d["results"]["time_primal"] = self.time_primal
        d["results"]["time_dual"] = self.time_dual
        d["results"]["time_solve"] = self.time_solve
        d["results"]["time_callback"] = self.time_callback
        d["results"]["num_callback"] = self.num_callback
        with open(fname,"w") as f:
            yaml.dump(d,f,default_flow_style=False)

class CuttingPlaneSocp:
    def __init__(self,mu_scenario,delta,p,k,gamma,timelimit=7200,mu=None,rho=None):
        self.mu_scenario = mu_scenario
        self.delta = delta
        self.p = p
        self.k = k
        self.gamma = gamma
        self.num_stock = self.mu_scenario.shape[1]
        self.num_scenario = self.mu_scenario.shape[0]
        self.weighted_mu_scenario = np.array([self.p[s]*self.mu_scenario[s,:] for s in range(self.num_scenario)])
        self.num_callback = 0
        self.time_callback = 0
        self.r_tol = 10**(-7)#*(1-self.delta)
        self.timelimit = timelimit
        self.rho=rho
        self.mu=mu
        if rho != None:
            self.mu_bar = calc_minumu_return(mu=mu,rho=rho,k=self.k)
    def check_feasibility(self,x_tmp,r_tmp,y_tmp):
        S_star =[]
        a  = time.time()
        x_vec = np.array([x_tmp[i] for i in range(self.num_stock) ])
        v_vec = -np.dot(self.mu_scenario,x_vec)-y_tmp
        S_star = [s for s in range(self.num_scenario) if v_vec[s] > 0]
        r_hat = sum(self.p[s]*(-sum(self.mu_scenario[s,i]*x_tmp[i] for i in x_tmp)-y_tmp) for s in S_star)
        return(S_star,r_hat)

    def solve(self,logfilename):
        start_all = time.time()
        start_modeling = time.time()
        self.model = grb.Model("PrimalSocpCVaRMinimization")
        def add_cutting_plane(model,where):
            if where  == grb.GRB.callback.MIPSOL:
                self.num_callback += 1
                start_callback = time.time()
                x_tmp = {}
                x_var = {}
                for var in model._vars:
                    if "x" in var.VarName:
                        i = int(var.VarName.split("_")[-1])
                        x_tmp[i] = model.cbGetSolution(var)
                        x_var[i] = var
                    if "r" in var.VarName:
                        r_tmp = model.cbGetSolution(var)
                        r_var = var
                    if "y" in var.VarName:
                        y_tmp = model.cbGetSolution(var)
                        y_var = var
                start=time.time()
                S_star, r_hat = self.check_feasibility(x_tmp = x_tmp,
                                                        r_tmp=r_tmp,
                                                        y_tmp=y_tmp)
                if r_tmp-r_hat >= -self.r_tol:
                    pass
                else:
                    A_vec = np.sum([self.weighted_mu_scenario[s,:] for s in S_star],axis=0)
                    b = np.sum([self.p[s] for s in S_star])
                    model.cbLazy(r_var >= grb.quicksum(-A_vec[i]*x_var[i] for i in range(self.num_stock))-b*y_var)
                    self.time_callback += time.time()-start_callback
        x = {}
        theta = {}
        z = {}
        THETA = {}
        Z = {}
        for i in range(self.num_stock):
            x[i] = self.model.addVar(vtype="C",name="x_"+str(i),lb=0,ub=1)
            theta[i] = self.model.addVar(vtype="C",name="theta_"+str(i),lb=0)
            z[i] = self.model.addVar(vtype="B",name="z_"+str(i))
            THETA[i] = self.model.addVar(vtype="C",ub=0.5)
            Z[i] = self.model.addVar(vtype="C",lb=0)
        y = self.model.addVar(vtype="C",name="y",lb=-grb.GRB.INFINITY)
        r = self.model.addVar(vtype="C",name="r",lb=0)
        self.model.update()
        self.model.addConstr(grb.quicksum(x[i] for i in range(self.num_stock))==1)
        self.model.addConstr(grb.quicksum(z[i] for i in range(self.num_stock))==self.k)
        for i in range(self.num_stock):
            self.model.addConstr(Z[i] == z[i]/2 + theta[i]/2)
            self.model.addConstr(THETA[i] == z[i]/2 - theta[i]/2)
            self.model.addConstr(x[i]*x[i] +THETA[i]*THETA[i] <= Z[i]*Z[i])
        A_vec = np.sum([self.weighted_mu_scenario[s,:] for s in range(self.num_scenario)],axis=0)
        b = np.sum([self.p[s] for s in range(self.num_scenario)])
        if self.mu_bar != None:
            self.model.addConstr(grb.quicksum(self.mu[i]*x[i] for i in range(self.num_stock))>=self.mu_bar)
        self.model.addConstr(r>=grb.quicksum(-A_vec[i]*x[i] for i in range(self.num_stock))-b*y)
        self.model.setObjective(grb.quicksum(theta[i] for i in range(self.num_stock))/(2*self.gamma)+y+r/(1-self.delta))
        self.model._vars = self.model.getVars()
        self.model.update()
        self.model.params.LazyConstraints = 1
        self.model.params.LogFile = logfilename
        self.model.params.NumericFocus = 3
        self.model.params.TIME_LIMIT = self.timelimit
        self.model.update()
        self.time_modeling = time.time() - start_modeling
        start_solve = time.time()
        self.model.optimize(add_cutting_plane)
        self.time_solve = time.time() -start_solve
        self.time_all = time.time()-start_all
    def get_optimal_sol(self):
        if self.model.Status == grb.GRB.OPTIMAL:
            x_opt = {}
            z_opt = {}
            for var in self.model.getVars():
                if 'z_' in var.VarName:
                    i = int(var.VarName.split('_')[-1])
                    z_opt[i] = var.X
                if 'x_' in var.VarName:
                    i = int(var.VarName.split('_')[-1])
                    x_opt[i] = var.X
            return(x_opt,z_opt)
        else:
            return(None)
    def get_optimal_val(self):
        if self.model.Status == grb.GRB.OPTIMAL:
            return(self.model.objVal)
        else:
            return(None)
    def output_results(self,fname):
        d = {}
        d["params"] = {}
        d["params"]["gamma"] = float(self.gamma)
        d["params"]["alpha"] = float(self.gamma*np.sqrt(self.num_stock))
        d["params"]["num_scenario"] = int(self.num_scenario)
        d["params"]["num_stock"] = int(self.num_stock)
        d["params"]["delta"] = float(self.delta)
        d["params"]["rho"] = float(self.rho)
        d["params"]["k"] = int(self.k)
        d["results"] = {}
        if self.model.Status == grb.GRB.OPTIMAL:
            x_opt, z_opt = self.get_optimal_sol()
            d["results"]["status"] = "optimal"
            d["results"]["x_opt"] = {}
            for i in x_opt:
                d["results"]["x_opt"][i] = float(x_opt[i]*z_opt[i])
            d["results"]["opt_val"] = self.get_optimal_val()
        elif self.model.Status == grb.GRB.TIME_LIMIT:
            d["results"]["status"] = "timelimit"
            d["results"]["obj_bst"] = self.model.ObjVal
            d["results"]["obj_bnd"] = self.model.ObjBound
            d["results"]["gap"] = self.model.MIPGap
        d["results"]["time_all"] = self.time_all
        d["results"]["time_model"] = self.time_modeling
        d["results"]["time_solve"] = self.time_solve
        d["results"]["time_callback"] = self.time_callback
        d["results"]["num_callback"] = self.num_callback
        with open(fname,"w") as f:
            yaml.dump(d,f,default_flow_style=False)

class CuttingPlaneBigM:
    def __init__(self,mu_scenario,delta,p,k,gamma,timelimit=7200,mu=None,rho=None):
        self.mu_scenario = mu_scenario
        self.delta = delta
        self.p = p
        self.k = k
        self.gamma = gamma
        self.num_stock = self.mu_scenario.shape[1]
        self.num_scenario = self.mu_scenario.shape[0]
        self.weighted_mu_scenario = np.array([self.p[s]*self.mu_scenario[s,:] for s in range(self.num_scenario)])
        self.num_callback = 0
        self.time_callback = 0
        self.r_tol = 10**(-5)*(1-self.delta)
        self.timelimit = timelimit
        self.rho=rho
        self.mu=mu
        if rho != None:
            self.mu_bar = calc_minumu_return(mu=mu,rho=rho,k=self.k)
    def check_feasibility(self,x_tmp,r_tmp,y_tmp):
        S_star =[]
        a  = time.time()
        x_vec = np.array([x_tmp[i] for i in range(self.num_stock) ])
        v_vec = -np.dot(self.mu_scenario,x_vec)-y_tmp
        S_star = [s for s in range(self.num_scenario) if v_vec[s] > 0]
        r_hat = sum(self.p[s]*(-sum(self.mu_scenario[s,i]*x_tmp[i] for i in x_tmp)-y_tmp) for s in S_star)
        return(S_star,r_hat)

    def solve(self,logfilename):
        start_all = time.time()
        start_modeling = time.time()
        self.model = grb.Model("PrimalSocpCVaRMinimization")

        def add_cutting_plane(model,where):
            if where  == grb.GRB.callback.MIPSOL:
                self.num_callback += 1
                start_callback = time.time()
                x_tmp = {}
                x_var = {}
                for var in model._vars:
                    if "x" in var.VarName:
                        i = int(var.VarName.split("_")[-1])
                        x_tmp[i] = model.cbGetSolution(var)
                        x_var[i] = var
                    if "r" in var.VarName:
                        r_tmp = model.cbGetSolution(var)
                        r_var = var
                    if "y" in var.VarName:
                        y_tmp = model.cbGetSolution(var)
                        y_var = var
                start = time.time()
                S_star, r_hat = self.check_feasibility(x_tmp = x_tmp,
                                                        r_tmp=r_tmp,
                                                        y_tmp=y_tmp)
                if r_tmp-r_hat >= -self.r_tol:
                    pass
                else:
                    A_vec = np.sum([self.weighted_mu_scenario[s,:] for s in S_star],axis=0)
                    b = np.sum([self.p[s] for s in S_star])
                    model.cbLazy(r_var >= grb.quicksum(-A_vec[i]*x_var[i] for i in range(self.num_stock))-b*y_var)
                    self.time_callback += time.time()-start_callback
        x = {}
        q = {}
        z = {}
        for i in range(self.num_stock):
            x[i] = self.model.addVar(vtype="C",name="x_"+str(i),lb=0,ub=1)
            z[i] = self.model.addVar(vtype="B",name="z_"+str(i))
        y = self.model.addVar(vtype="C",name="y",lb=-grb.GRB.INFINITY)
        r = self.model.addVar(vtype="C",name="r",lb=0)
        self.model.update()
        self.model.addConstr(grb.quicksum(x[i] for i in range(self.num_stock))==1)
        self.model.addConstr(grb.quicksum(z[i] for i in range(self.num_stock))==self.k)
        for i in range(self.num_stock):
            self.model.addConstr(x[i]<= z[i])
        A_vec = np.sum([self.weighted_mu_scenario[s,:] for s in range(self.num_scenario)],axis=0)
        b = np.sum([self.p[s] for s in range(self.num_scenario)])
        self.model.addConstr(r>=grb.quicksum(-A_vec[i]*x[i] for i in range(self.num_stock))-b*y)
        if self.mu_bar != None:
            self.model.addConstr(grb.quicksum(self.mu[i]*x[i] for i in x) >= self.mu_bar)
        self.model.setObjective(grb.quicksum(x[i]*x[i] for i in range(self.num_stock))/(2*self.gamma)+y+r/(1-self.delta))
        self.model._vars = self.model.getVars()
        self.model.update()
        self.model.params.LazyConstraints = 1
        self.model.params.LogFile = logfilename
        self.model.params.TIME_LIMIT = self.timelimit
        self.model.update()
        self.time_modeling = time.time() - start_modeling
        start_solve = time.time()
        self.model.optimize(add_cutting_plane)
        self.time_solve = time.time() -start_solve
        self.time_all = time.time()-start_all
    def get_optimal_sol(self):
        if self.model.Status == grb.GRB.OPTIMAL:
            x_opt = {}
            z_opt = {}
            for var in self.model.getVars():
                if 'z_' in var.VarName:
                    i = int(var.VarName.split('_')[-1])
                    z_opt[i] = var.X
                if 'x_' in var.VarName:
                    i = int(var.VarName.split('_')[-1])
                    x_opt[i] = var.X
            return(x_opt,z_opt)
        else:
            return(None)
    def get_optimal_val(self):
        if self.model.Status == grb.GRB.OPTIMAL:
            return(self.model.objVal)
        else:
            return(None)
    def output_results(self,fname):
        d = {}
        d["params"] = {}
        d["params"]["gamma"] = float(self.gamma)
        d["params"]["alpha"] = float(self.gamma*np.sqrt(self.num_stock))
        d["params"]["num_scenario"] = int(self.num_scenario)
        d["params"]["num_stock"] = int(self.num_stock)
        d["params"]["delta"] = float(self.delta)
        d["params"]["rho"] = float(self.rho)
        d["params"]["k"] = int(self.k)
        d["results"] = {}
        if self.model.Status == grb.GRB.OPTIMAL:
            x_opt, z_opt = self.get_optimal_sol()
            d["results"]["status"] = "optimal"
            d["results"]["x_opt"] = {}
            for i in x_opt:
                d["results"]["x_opt"][i] = float(x_opt[i]*z_opt[i])
            d["results"]["opt_val"] = self.get_optimal_val()
        elif self.model.Status == grb.GRB.TIME_LIMIT:
            d["results"]["status"] = "timelimit"
            d["results"]["obj_bst"] = self.model.ObjVal
            d["results"]["obj_bnd"] = self.model.ObjBound
            d["results"]["gap"] = self.model.MIPGap
        d["results"]["time_all"] = self.time_all
        d["results"]["time_model"] = self.time_modeling
        d["results"]["time_solve"] = self.time_solve
        d["results"]["time_callback"] = self.time_callback
        d["results"]["num_callback"] = self.num_callback
        with open(fname,"w") as f:
            yaml.dump(d,f,default_flow_style=False)
