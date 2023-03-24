# low-rank-path-following
Low-rank path-following code

The file main.py allows the user to choose whether to load existing problem data or create new ones.
The data are stored in a dictionary contained pickled file "file_name.pkl".
The default structure of the dictionary is 

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
                'IPM_runtime'                         : {str(d) : [] for d in stepsizes}}


1a. LOAD_DATA is set to False.
    The user is asked to enter a name for the file.
    The a pickled file with the desired name is created in the experiments folder.
    The user is asked to enter the desired number of instances to create and test.
    The user is asked to decide whether to creat a Time-Varying Max Cur Relaxation problem or a random SDP (see create_problem_data.py).
    The user is asked to enter the dimensions of the problem.

1b. LOAD_DATA is set to True.
    DATA_FILE is string with the name of a .pkl file contained in the experiments folder, e.g. 'SIAM_DATA'.
    The data dictionary is load, potentially already populated with the experiments results.

2.  RUN_PATH_FOLL, RUN_MOSEK, RUN_SCS are boolean to select which procedures to use.
3.  SAVE_EXP requires to save the results on the pkl file.
4.  PLOT_EXP requires to plot the runtime and the residual accuracy of the results of the experiments.

The second-to-last block of inputs for the constructor of the class _Experiments corresponds to the parameters of the path-folllowing procedure.
The last block of inputs for the function exp._Experiments corresponds to the accuracy parameters of the procedures.

The solutions for the TV-SDP are contained in the list attribute _primal_solutions_list of the _PathFollowing class.

For a detailed explaination of the path-following algorithm we refer to https://arxiv.org/pdf/2210.08387.pdf.