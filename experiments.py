import numpy as np
import datetime
import time 
import pickle

import create_problem_data  as cpd
import mosek_ipm_solver as mis 
import plotting_tools as plot
import path_following as pfl  
import scs_tracker as scs
import ipm_tracker as ipm 

class _Experiments:

    def __init__(self, 
                initial_time          : np.float,
                final_time            : np.float, 
                stepsize  : np.float,
                residual_tolerance    : np.float,
                gamma_1               : np.float,
                gamma_2               : np.float,
                STEPSIZE_TUNING       : np.bool, 
                FOLLOW_GRID           : np.bool,

                init_point_precision  : np.float,
                scs_tolerance         : np.float,
                ipm_tolerance         : np.float,

                DATA_FILE             : np.str, 
                LOAD_DATA             : np.bool,  
                SAVE_EXP              : np.bool,
                RUN_PATH_FOLL         : np.bool,
                RUN_MOSEK             : np.bool,
                RUN_SCS               : np.bool,
                PLOT_EXP              : np.bool,
                ):
        
        self.initial_time         = initial_time    
        self.final_time           = final_time 
        self.stepsize             = stepsize
        self.res_tolerance        = residual_tolerance
        self.gamma_1              = gamma_1
        self.gamma_2              = gamma_2
        self.STEPSIZE_TUNING      = STEPSIZE_TUNING  
        self.FOLLOW_GRID          = FOLLOW_GRID

        self.init_point_precision = init_point_precision  
        self.scs_tolerance        = scs_tolerance  
        self.ipm_tolerance        = ipm_tolerance  

        self.DATA_FILE            = DATA_FILE 
        self.LOAD_DATA            = LOAD_DATA  
        self.SAVE_EXP             = SAVE_EXP 
        self.RUN_MOSEK            = RUN_MOSEK
        self.RUN_SCS              = RUN_SCS
        self.RUN_PATH_FOLL        = RUN_PATH_FOLL
        self.PLOT_EXP             = PLOT_EXP

        self.exp_dict             = {}

    def create_exp_file(self,):

            stepsizes = [10**k  for k in range(1, 7)] 
            exp_dict  = {'exp_data'                               : {'A_init' : [], 'A_pert' : [], 
                                                                     'b_init' : [], 'b_pert' : [], 
                                                                     'C_init' : [], 'C_pert' : []},  
                            'PC_residuals'                        : {str(d) : [] for d in stepsizes}, 
                            'PC_SDP_residuals'                    : {str(d) : [] for d in stepsizes}, 
                            'PC_runtime'                          : {str(d) : [] for d in stepsizes}, 
                            'SCS_residuals'                       : {str(d) : [] for d in stepsizes},
                            'SCS_runtime'                         : {str(d) : [] for d in stepsizes}, 
                            'IPM_residuals'                       : {str(d) : [] for d in stepsizes},
                            'IPM_runtime'                         : {str(d) : [] for d in stepsizes},
                            'ranks'                               : []}

            dt = datetime.datetime.now()
            dt_string = '['+str(dt)[11:16]+' '+str(dt)[5:10]+'] '

            file_name = str(input("Enter file name:"))
            self.DATA_FILE = dt_string + file_name + '.pkl'

            f = open('experiments/' + self.DATA_FILE, 'wb')
            pickle.dump(exp_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.close()

            return exp_dict

    def load_exp_file(self, file_name):
        
            f = open('experiments/' + file_name + '.pkl', 'rb')
            exp_dict = pickle.load(f) 
            f.close()

            return exp_dict

    def get_functions_from_data(self, exp_dict, data_index):

        k=data_index

        def A(time: np.float): return exp_dict['exp_data']['A_init'][k] + time*exp_dict['exp_data']['A_pert'][k]
        def b(time: np.float): return exp_dict['exp_data']['b_init'][k] + time*exp_dict['exp_data']['b_pert'][k]
        def C(time: np.float): return exp_dict['exp_data']['C_init'][k] + time*exp_dict['exp_data']['C_pert'][k]

        return A, b, C 

    def time_estimate(self,running_times, data_index, NR_INSTANCES):

        if data_index > 0:
            mean_time = np.mean(running_times) 
            minutes = np.str(np.int(np.divmod((NR_INSTANCES-data_index)*mean_time,60)[0]))
            seconds = np.str(np.int(np.divmod((NR_INSTANCES-data_index)*mean_time,60)[1]))
            if len(seconds)==1: seconds = '0'+seconds
            centiseconds = np.str(mean_time-np.floor(mean_time))
            print('Instance',data_index,'/',NR_INSTANCES," | time to termination: "+minutes+":"+seconds+":"+centiseconds[2:4])  

    def run_experiments(self):
        
        t_0 = self.initial_time
        t_f = self.final_time

        if self.LOAD_DATA:
            
            self.exp_dict = self.load_exp_file(self.DATA_FILE) 
            NR_INSTANCES = len(self.exp_dict['exp_data']['A_init']) 

        else: 

            self.exp_dict = self.create_exp_file()
            NR_INSTANCES =  int(input("Enter number of instances: "))

            problem_type = str(input("Select problem type:\n'M' --> TV-MaxCut Relaxation\n'R' --> random TV-SDP\n"))
            problem = cpd._ProblemCreator()
            if problem_type == 'M': 
                n = int(input("Enter problem dimension n: "))
            if problem_type == 'R':
                n = int(input("Enter problem dimension n: "))
                m = int(input("Enter problem dimension m: "))
                     
            for inst in range(NR_INSTANCES):

                if problem_type == 'M': 
                    A, b, C = problem._create_MaxCut(n)

                if problem_type == 'R': 
                    A, b, C = problem._create_random_problem(n,m,sparsity_coefficient=0.5)
                    
                self.exp_dict['exp_data']['A_init'].append(A(t_0)) 
                self.exp_dict['exp_data']['A_pert'].append(A(t_f)-A(t_0))  
                self.exp_dict['exp_data']['b_init'].append(b(t_0)) 
                self.exp_dict['exp_data']['b_pert'].append(b(t_f)-b(t_0))  
                self.exp_dict['exp_data']['C_init'].append(C(t_0)) 
                self.exp_dict['exp_data']['C_pert'].append(C(t_f)-C(t_0)) 
            
        if self.RUN_PATH_FOLL:

            print("Running BM tracker")
             
            running_times = []
            
            for data_index in range(NR_INSTANCES):

                self.time_estimate(running_times, data_index, NR_INSTANCES) 
                before = time.time()
                
                A, b, C = self.get_functions_from_data(self.exp_dict, data_index)
                Y_0, rank, lam_0, init_run_time  = mis._get_initial_point(A=A(t_0), b=b(t_0), C=C(t_0), rel_gap_tol=self.init_point_precision) 

                pathfoll = pfl._PathFollowing(n=np.shape(A(0))[1], m=np.shape(A(0))[0], r=rank)  

                pathfoll.run(A, b, C, Y_0, lam_0, 
                            initial_time       = t_0, 
                            final_time         = t_f, 
                            initial_stepsize   = self.stepsize, 
                            gamma_1            = self.gamma_1, 
                            gamma_2            = self.gamma_2,
                            residual_tolerance = self.res_tolerance,
                            STEPSIZE_TUNING    = self.STEPSIZE_TUNING,
                            FOLLOW_GRID        = self.FOLLOW_GRID) 

                sub = int(1/self.stepsize)
                self.exp_dict['PC_residuals'][str(sub)].append(pathfoll._PC_residuals)
                self.exp_dict['PC_SDP_residuals'][str(sub)].append(pathfoll._PC_SDP_residuals)
                self.exp_dict['PC_runtime'  ][str(sub)].append(pathfoll._PC_runtime+init_run_time) 
                
                running_times.append(time.time() - before)

        if self.RUN_MOSEK:

            print("Running MOSEK tracker")

            running_times = []
            
            for data_index in range(NR_INSTANCES):
            
                self.time_estimate(running_times, data_index, NR_INSTANCES) 
                before = time.time()
                
                A, b, C = self.get_functions_from_data(self.exp_dict, data_index)

                ipm_track = ipm._IPM_tracker()
                ipm_track.run(A, b, C, 
                            initial_time  = t_0,
                            final_time    = t_f,
                            stepsize      = self.stepsize, 
                            ipm_tolerance = self.ipm_tolerance) 

                sub = int(1/self.stepsize)
                self.exp_dict['IPM_residuals'][str(sub)].append(ipm_track._IPM_residuals)
                self.exp_dict['IPM_runtime'  ][str(sub)].append(ipm_track._IPM_runtime)  

                running_times.append(time.time() - before) 
                 
        if self.RUN_SCS:

            print("Running SCS tracker")

            running_times = []
            
            for data_index in range(NR_INSTANCES): 

                self.time_estimate(running_times, data_index, NR_INSTANCES) 
                before = time.time()
                
                A, b, C = self.get_functions_from_data(self.exp_dict, data_index)

                scs_track = scs._SCS_tracker() 
                scs_track.run(A, b, C, 
                            initial_time  = t_0,
                            final_time    = t_f,
                            stepsize      = self.stepsize, 
                            scs_tolerance = self.scs_tolerance) 

                sub = int(1/self.stepsize)
                self.exp_dict['SCS_residuals'][str(sub)].append(scs_track._SCS_residuals)
                self.exp_dict['SCS_runtime'  ][str(sub)].append(scs_track._SCS_runtime)  
                
                running_times.append(time.time() - before)
        
        if self.SAVE_EXP:
             
            f = open('experiments/'+self.DATA_FILE,'wb')
            pickle.dump(self.exp_dict, f, pickle.HIGHEST_PROTOCOL)
            f.close

        if self.PLOT_EXP :

            plotools = plot._Plottingtools(PLOT_DEGEN=False)
            plotools.plot_residual_VS_subdivision(self.exp_dict,self.load_exp_file('rank_deg'))
            plotools.plot_runtime_VS_subdivision(self.exp_dict)
            # plotools.plot_runtime_VS_grids(self.exp_dict)
            # plotools.plot_residual_VS_parameter(self.exp_dict)