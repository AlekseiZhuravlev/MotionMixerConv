#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
# from datasets.dataset_h36m import Datasets
#from utils.data_utils import define_actions


def mpjpe_error(batch_pred,batch_gt): 
    
    batch_pred= batch_pred.contiguous().view(-1,3)
    batch_gt=batch_gt.contiguous().view(-1,3)

    return torch.mean(torch.norm(batch_gt-batch_pred,2,1))


def define_actions(action):
    """
    Define the list of actions we are using.

    Args
      action: String with the passed action. Could be "all"
    Returns
      actions: List of strings of actions
    Raises
      ValueError if the action is not included in H3.6M
    """

    actions = ["walking", "eating", "smoking", "discussion", "directions",
               "greeting", "phoning", "posing", "purchases", "sitting",
               "sittingdown", "takingphoto", "waiting", "walkingdog",
               "walkingtogether"]
    if action in actions:
        return [action]

    if action == "all":
        return actions

    if action == "all_srnn":
        return ["walking", "eating", "smoking", "discussion"]

    raise (ValueError, "Unrecognized action: %d" % action)
    
def create_pose(ax,plots,vals,pred=True,update=False):

            
    
    # h36m 32 joints(full)
    connect = [
            (1, 2), (2, 3), (3, 4), (4, 5),
            (6, 7), (7, 8), (8, 9), (9, 10),
            (0, 1), (0, 6),
            (6, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 22),
            (1, 25), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30),
            (24, 25), (24, 17),
            (24, 14), (14, 15)
    ]
    LR = [
            False, True, True, True, True,
            True, False, False, False, False,
            False, True, True, True, True,
            True, True, False, False, False,
            False, False, False, False, True,
            False, True, True, True, True,
            True, True
    ]  


# Start and endpoints of our representation
    I   = np.array([touple[0] for touple in connect])
    J   = np.array([touple[1] for touple in connect])
# Left / right indicator
    LR  = np.array([LR[a] or LR[b] for a,b in connect])
    if pred:
        lcolor = "#9b59b6"
        rcolor = "#2ecc71"
    else:
        lcolor = "#8e8e8e"
        rcolor = "#383838"

    for i in np.arange( len(I)):
        x = np.array( [vals[I[i], 0], vals[J[i], 0]] )
        z = np.array( [vals[I[i], 1], vals[J[i], 1]] )
        y = np.array( [vals[I[i], 2], vals[J[i], 2]] )
        if not update:

            if i ==0:
                plots.append(ax.plot(x, y, z, lw=2,linestyle='--' ,c=lcolor if LR[i] else rcolor,label=['GT' if not pred else 'Pred']))
            else:
                plots.append(ax.plot(x, y, z, lw=2,linestyle='--', c=lcolor if LR[i] else rcolor))

        elif update:
            plots[i][0].set_xdata(x)
            plots[i][0].set_ydata(y)
            plots[i][0].set_3d_properties(z)
            plots[i][0].set_color(lcolor if LR[i] else rcolor)
    
    return plots
   # ax.legend(loc='lower left')


# In[11]:


def update(num,data_gt,data_pred,plots_gt,plots_pred,fig,ax):
    
    gt_vals=data_gt[num]
    pred_vals=data_pred[num]
    plots_gt=create_pose(ax,plots_gt,gt_vals,pred=False,update=True)
    plots_pred=create_pose(ax,plots_pred,pred_vals,pred=True,update=True)
    
    

    
    
    r = 0.75
    xroot, zroot, yroot = gt_vals[0,0], gt_vals[0,1], gt_vals[0,2]
    ax.set_xlim3d([-r+xroot, r+xroot])
    ax.set_ylim3d([-r+yroot, r+yroot])
    ax.set_zlim3d([-r+zroot, r+zroot])
    #ax.set_title('pose at time frame: '+str(num))
    #ax.set_aspect('equal')
 
    return plots_gt,plots_pred
    





#%%


def visualize(input_n,output_n,visualize_from,path,modello,device,n_viz,skip_rate,actions,encoding ='dct'):
    
    import random 
    actions=define_actions(actions)
    
    for action in actions:
    
        if visualize_from=='train':
            loader=Datasets(path,input_n,output_n,skip_rate, split=0,actions=[action])
        elif visualize_from=='validation':
            loader=Datasets(path,input_n,output_n,skip_rate, split=1,actions=[action])
        elif visualize_from=='test':
            loader=Datasets(path,input_n,output_n,skip_rate, split=2,actions=[action])
            
        dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                        26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                        46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                        75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])
      # joints at same loc
        joint_to_ignore = np.array([16, 20, 23, 24, 28, 31])
        index_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
        joint_equal = np.array([13, 19, 22, 13, 27, 30])
        index_to_equal = np.concatenate((joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))
            
            
        loader = DataLoader(
        loader,
        batch_size=256,
        shuffle = False, # for comparable visualizations with other models
        num_workers=0)       
        
            
    
        for cnt,batch in enumerate(loader): 
            batch = batch.to(device) 
            
            all_joints_seq=batch.clone()[:, input_n:input_n+output_n,:]
            
            sequences_train=batch[:, 0:input_n, dim_used].view(-1,input_n,len(dim_used))
            sequences_gt=batch[:, input_n:input_n+output_n, :]
            
            sequences_predict=modello(sequences_train).contiguous().view(-1,output_n,len(dim_used))
            
            all_joints_seq[:,:,dim_used] = sequences_predict
            
            all_joints_seq[:,:,index_to_ignore] = all_joints_seq[:,:,index_to_equal]
            
            
            all_joints_seq=all_joints_seq.view(-1,output_n,32,3)
            
            sequences_gt=sequences_gt.view(-1,output_n,32,3)
            
            loss=mpjpe_error(all_joints_seq,sequences_gt)# # both must have format (batch,T,V,C)
    
            
    
            data_pred=torch.squeeze(all_joints_seq,0).cpu().data.numpy()/1000 # in meters
            data_gt=torch.squeeze(sequences_gt,0).cpu().data.numpy()/1000
            
            i = random.randint(1,256)
            
            data_pred = data_pred [i]
            data_gt = data_gt [i]
            
            #print (data_gt.shape,data_pred.shape)
    
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.view_init(elev=20, azim=-40)
            vals = np.zeros((32, 3)) # or joints_to_consider
            gt_plots=[]
            pred_plots=[]
    
            gt_plots=create_pose(ax,gt_plots,vals,pred=False,update=False)
            pred_plots=create_pose(ax,pred_plots,vals,pred=True,update=False)
    
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.legend(loc='lower left')
    
    
    
            ax.set_xlim3d([-1, 1.5])
            ax.set_xlabel('X')
    
            ax.set_ylim3d([-1, 1.5])
            ax.set_ylabel('Y')
    
            ax.set_zlim3d([0.0, 1.5])
            ax.set_zlabel('Z')
            ax.set_title('loss in mm is: '+str(round(loss.item(),4))+' for action : '+str(action)+' for '+str(output_n)+' frames')
    
            line_anim = animation.FuncAnimation(fig, update, output_n, fargs=(data_gt,data_pred,gt_plots,pred_plots,
                                                                       fig,ax),interval=70, blit=False)
            plt.show()
            
            line_anim.save('./visualizations/pred{}/human_viz{}.gif'.format (25,i),writer='pillow')
    
            
            if cnt==n_viz-1:
                break
            
            
            
            
            