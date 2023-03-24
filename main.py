import experiments as exp 
  
experiments = exp._Experiments(LOAD_DATA             = False,          # controls whether to load old data or create new ones instead
                               DATA_FILE             = 'SIAM_DATA',    # name of the data file to load 
                               
                               RUN_PATH_FOLL         = False,          # controls which procedure to run
                               RUN_MOSEK             = False,
                               RUN_SCS               = False,

                               SAVE_EXP              = False,          # controls whether to save the experiments
                               PLOT_EXP              = True,           # controls whether to plot the experiments

                               initial_time          = 0,              # initial time for the path following
                               final_time            = 1,              # final time for the path following 
                               stepsize              = 0.1,            # value must be contained in [1.e-k  for k in range(1, 7)]
                               gamma_1               = 0.5,
                               gamma_2               = 1.5,
                               STEPSIZE_TUNING       = False,          # activates the residual stepsize tuning
                               FOLLOW_GRID           = False,          # request the path-following tracking to followa resolution grid

                               residual_tolerance    = 1.e-05,         # Residual accuracy parameter for the relative gap tolerance for the intial point of the path-following 
                               init_point_precision  = 1e-14,          # Relative gap tolerance for the intial point of the path-following (mosek.dparam.intpnt_co_tol_rel_gap https://docs.mosek.com/latest/opt-server/parameters.html)
                               ipm_tolerance         = 1e-8,           # Relative gap tolerance for the IPM tracking (mosek.dparam.intpnt_co_tol_rel_gap https://docs.mosek.com/latest/opt-server/parameters.html)
                               scs_tolerance         = 1e-6)           # Accuracy parameters for SCS solver https://www.cvxgrp.org/scs/algorithm/index.html#termination

experiments.run_experiments()