from dataclasses import asdict
from traceback import print_tb
from operator import add

import plotly.graph_objects as go 
import matplotlib.pyplot as plt
import numpy as np 
import pickle

class _Plottingtools:

    def __init__(self, PLOT_DEGEN: np.bool) -> None:
        
        self.stepsizes = [10**k  for k in range(1, 7)] 
        self.gridsizes = [20,40,60,80,100] 
        self.PLOT_DEGEN = PLOT_DEGEN

        self.green    = '#3D9970'
        self.lime     = '#27E625'
        self.red      = '#FF4136'
        self.orange   = '#FF851B'
        self.carrot   = '#FFC26A'
        self.bordeaux = '#800020'

    def plot_residual_VS_subdivision(self, exp_dict, degen_dic): 
        
        stepsizes = self.stepsizes
        NR_INSTANCES = len(exp_dict['exp_data']['A_init'])
        
        PC_residuals     = [np.mean(x) for sub in stepsizes for x in exp_dict['PC_residuals'][str(sub)]]
        PC_SDP_residuals = [np.mean(x) for sub in stepsizes for x in exp_dict['PC_SDP_residuals'][str(sub)]]
        SCS_residuals    = [np.mean(x) for sub in stepsizes for x in exp_dict['SCS_residuals'][str(sub)]]  
        IPM_residuals    = [np.mean(x) for sub in stepsizes for x in exp_dict['IPM_residuals'][str(sub)]]  

        PC_subdivisions     = ["{:.0e}".format(1/sub) for sub in stepsizes for x in range(NR_INSTANCES) if not exp_dict['PC_residuals'][str(sub)]==[]]
        PC_SDP_subdivisions = ["{:.0e}".format(1/sub) for sub in stepsizes for x in range(NR_INSTANCES) if not exp_dict['PC_SDP_residuals'][str(sub)]==[]]
        SCS_subdivisions    = ["{:.0e}".format(1/sub) for sub in stepsizes for x in range(NR_INSTANCES) if not exp_dict['SCS_residuals'][str(sub)]==[]]
        IPM_subdivisions    = ["{:.0e}".format(1/sub) for sub in stepsizes for x in range(NR_INSTANCES) if not exp_dict['IPM_residuals'][str(sub)]==[]]
        
        fig = go.Figure() 

        fig.add_trace(go.Box(
            x            = IPM_subdivisions,
            y            = IPM_residuals,
            name         = r'$\text{IPM residual}$', 
            marker_color = self.bordeaux,
            boxpoints    = False
            ))

        fig.add_trace(go.Box(
            x            = SCS_subdivisions,
            y            = SCS_residuals,
            name         = r'$\text{SCS residual}$', 
            marker_color = self.orange,
            boxpoints    = False
            )) 

        fig.add_trace(go.Box(
            x            = PC_SDP_subdivisions,
            y            = PC_SDP_residuals,
            name         = r'$\text{PC}\ \ \ \text{residual}$', 
            marker_color = self.green,
            boxpoints    = False
            )) 

        if self.PLOT_DEGEN: 

            NR_INSTANCES = len(degen_dic['exp_data']['A_init'])
        
            PC_SDP_residuals_deg     = [np.mean(x) for sub in stepsizes for x in degen_dic['PC_SDP_residuals'][str(sub)]] 
            PC_SDP_subdivisions_deg  = ["{:.0e}".format(1/sub) for sub in stepsizes for x in range(NR_INSTANCES) if not degen_dic['PC_SDP_residuals'][str(sub)]==[]] 
            
            fig.add_trace(go.Scatter(
                x            = PC_SDP_subdivisions_deg,
                y            = PC_SDP_residuals_deg,
                mode         = 'markers',
                marker_color = self.lime,
                name         = r'$\text{rank changing instances}$', )) 

        fig.update_layout(
            width             = 810,
            height            = 500,
            yaxis_title       = r'$\text{residuals}$', 
            xaxis_title       = r'$\text{stepsize}$', 
            autosize          = True,
            font_family       = "Helvetica", 
            title_font_family = "Helvetica", 
            paper_bgcolor     = 'rgba(0,0,0,0)',
            plot_bgcolor      = 'rgba(0,0,0,0)',
            font=dict(size=18)
            ) 

        fig.update_xaxes(type="category", nticks=6, ticks="outside")
        fig.update_yaxes(type="log", ticks="outside",exponentformat = 'e')
        fig.show()
         
    def plot_runtime_VS_subdivision(self, exp_dict): 
        
        stepsizes = self.stepsizes
        NR_INSTANCES= len(exp_dict['exp_data']['A_init'])
 
        PC_runtimes      = [np.mean(x) for sub in stepsizes for x in exp_dict['PC_runtime'][str(sub)]]
        SCS_runtimes     = [np.mean(x) for sub in stepsizes for x in exp_dict['SCS_runtime'][str(sub)]]
        IPM_runtimes     = [np.mean(x) for sub in stepsizes for x in exp_dict['IPM_runtime'][str(sub)]]
        PC_subdivisions  = ["{:.0e}".format(1/sub) for sub in stepsizes for x in range(NR_INSTANCES) if not exp_dict['PC_runtime'][str(sub)]==[]]
        SCS_subdivisions = ["{:.0e}".format(1/sub) for sub in stepsizes for x in range(NR_INSTANCES) if not exp_dict['SCS_runtime'][str(sub)]==[]]
        IPM_subdivisions = ["{:.0e}".format(1/sub) for sub in stepsizes for x in range(NR_INSTANCES) if not exp_dict['IPM_runtime'][str(sub)]==[]]
        
        fig = go.Figure() 

        fig.add_trace(go.Box(
            x            = IPM_subdivisions,
            y            = IPM_runtimes,
            name         = r'$\text{IPM}$', 
            marker_color = self.bordeaux,
            boxpoints    = False
            ))

        fig.add_trace(go.Box(
            x            = SCS_subdivisions,
            y            = SCS_runtimes,
            name         = r'$\text{SCS}$', 
            marker_color = self.orange,
            boxpoints    = False
            ))

        fig.add_trace(go.Box(
            x            = PC_subdivisions,
            y            = PC_runtimes,
            name         = r'$\text{PC}$', 
            marker_color = self.green,
            boxpoints    = False
            ))

        fig.update_layout(
            width             = 810,
            height            = 500,
            yaxis_title       = r'$\text{runtime }[sec]$', 
            xaxis_title       = r'$\text{stepsize}$', 
            autosize          = True,
            font_family       = "Helvetica", 
            title_font_family = "Helvetica", 
            paper_bgcolor     = 'rgba(0,0,0,0)',
            plot_bgcolor      = 'rgba(0,0,0,0)',
            font=dict(size=18)
            ) 

        fig.update_xaxes(type="category", nticks=6, ticks="outside")
        fig.update_yaxes(type="log", ticks="outside",exponentformat = 'e')
        fig.show()
    
    def plot_runtime_VS_grids(self, exp_dict): 
        
        stepsizes = self.gridsizes
        NR_INSTANCES = len(exp_dict['exp_data']['A_init'])
        
        PC_runtimes      = [np.mean(x) for sub in stepsizes for x in exp_dict['PC_runtime'][str(sub)]]
        IPM_runtimes     = [np.mean(x) for sub in stepsizes for x in exp_dict['IPM_runtime'][str(sub)]]
        PC_subdivisions  = [sub for sub in stepsizes for x in range(NR_INSTANCES) if not exp_dict['PC_runtime'][str(sub)]==[]]
        IPM_subdivisions = [sub for sub in stepsizes for x in range(NR_INSTANCES) if not exp_dict['IPM_runtime'][str(sub)]==[]]
        
        fig = go.Figure() 

        fig.add_trace(go.Box(
            x            = IPM_subdivisions,
            y            = IPM_runtimes,
            name         = r'$\text{IPM}$', 
            marker_color = self.bordeaux,
            boxpoints    = False
            ))

        fig.add_trace(go.Box(
            x            = PC_subdivisions,
            y            = PC_runtimes,
            name         = r'$\text{PC}$', 
            marker_color = self.green,
            boxpoints    = False
            ))

        fig.update_layout(
            width             = 810,
            height            = 500,
            yaxis_title       = r'$\text{runtime }[sec]$', 
            xaxis_title       = r'$\text{# gridpoints}$', 
            autosize          = True,
            font_family       = "Helvetica", 
            title_font_family = "Helvetica", 
            paper_bgcolor     = 'rgba(0,0,0,0)',
            plot_bgcolor      = 'rgba(0,0,0,0)',
            font=dict(size=18)
            )

        fig.update_xaxes(type="category", nticks=6, ticks="outside")
        fig.update_yaxes(type="log", ticks="outside",exponentformat = 'e')
        fig.show()

    def plot_residual_VS_parameter(self, exp_dict): 
        
        NR_INSTANCES= len(exp_dict['exp_data']['A_init'])

        for sub in self.stepsizes:
            if not exp_dict['PC_residuals'][str(sub)] == []:
                for k in range(NR_INSTANCES):
                    
                    residuals = exp_dict['PC_residuals'][str(sub)][k]
                    l = len(residuals)
                    plt.plot([i/l for i in range(l)], residuals)

                plt.show()