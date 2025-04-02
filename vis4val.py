import os
import sys
import numpy as np
from scipy import spatial as ss
from pathlib import Path

import cv2
from misc.utils import hungarian,read_pred_and_gt,AverageMeter,AverageCategoryMeter

# dataset = 'NWPU'
# dataset = 'SHHA'
dataset = 'SHHB'
# dataRoot = '../ProcessedData/' + dataset
# temp = "C:/Users/rsriram3/ind_study/IIM/vis4val.py"
# dataRoot = '../ShanghaiTech Data/' + dataset
dataRoot = Path(r"C:\Users\rsriram3\Documents\ind_study\ShanghaiTech Data") / dataset
gt_file = dataRoot / 'val_gt_loc.txt'
# img_path = ori_data = dataRoot + '/images'
img_path = dataRoot / 'images'

# exp_name = './saved_exp_results/XXX_vis_results'
exp_name = Path('./saved_exp_results/XXX_vis_results')
# pred_file = 'NWPU_HR_Net_val.txt'
pred_path = Path('saved_exp_results/val_results')
# pred_file = Path("saved_exp_results/SHHB_VGG16_FPN_val.txt")
pred_file = pred_path / 'SHHB_VGG16_FPN_val.txt'

flagError = False
id_std = [i for i in range(3110,3610,1)]
id_std[59] = 3098

if not os.path.exists(exp_name):
    os.mkdir(exp_name)

def main():
    
    pred_data, gt_data = read_pred_and_gt(pred_file,gt_file)

    ## NEW ADDITION
    valid_ids = list(set(pred_data.keys()) & set(gt_data.keys()))

    for i_sample in valid_ids:
    # for i_sample in id_std:

        print(i_sample)        
        
        gt_p,pred_p,fn_gt_index,tp_pred_index,fp_pred_index,ap,ar= [],[],[],[],[],[],[]

        if gt_data[i_sample]['num'] ==0 and pred_data[i_sample]['num'] !=0:            
            pred_p =  pred_data[i_sample]['points']
            fp_pred_index = np.array(range(pred_p.shape[0]))
            ap = 0
            ar = 0

        if pred_data[i_sample]['num'] ==0 and gt_data[i_sample]['num'] !=0:
            gt_p = gt_data[i_sample]['points']
            fn_gt_index = np.array(range(gt_p.shape[0]))
            sigma_l = gt_data[i_sample]['sigma'][:,1]
            ap = 0
            ar = 0

        if gt_data[i_sample]['num'] !=0 and pred_data[i_sample]['num'] !=0:
            pred_p =  pred_data[i_sample]['points']    
            gt_p = gt_data[i_sample]['points']
            sigma_l = gt_data[i_sample]['sigma'][:,1]
            level = gt_data[i_sample]['level']        
        
            # dist
            dist_matrix = ss.distance_matrix(pred_p,gt_p,p=2)
            match_matrix = np.zeros(dist_matrix.shape,dtype=bool)
            for i_pred_p in range(pred_p.shape[0]):
                pred_dist = dist_matrix[i_pred_p,:]
                match_matrix[i_pred_p,:] = pred_dist<=sigma_l
                
            # hungarian outputs a match result, which may be not optimal. 
            # Nevertheless, the number of tp, fp, tn, fn are same under different match results
            # If you need the optimal result for visualzation, 
            # you may treat it as maximum flow problem. 
            tp, assign = hungarian(match_matrix)
            fn_gt_index = np.array(np.where(assign.sum(0)==0))[0]
            tp_pred_index = np.array(np.where(assign.sum(1)==1))[0]
            tp_gt_index = np.array(np.where(assign.sum(0)==1))[0]
            fp_pred_index = np.array(np.where(assign.sum(1)==0))[0]



            pre = tp_pred_index.shape[0]/(tp_pred_index.shape[0]+fp_pred_index.shape[0]+1e-20)
            rec = tp_pred_index.shape[0]/(tp_pred_index.shape[0]+fn_gt_index.shape[0]+1e-20)

        img_file = img_path / f"{i_sample:04d}.jpg"
        if not img_file.exists():
            print(f"Image {img_file} not found. Skipping {img_file}")
            continue
        img = cv2.imread(str(img_file))#bgr
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        if img is None:
            print(f"Failed to load {img_file}")
            continue
        
        point_r_value = 5
        thickness = 3
        if gt_data[i_sample]['num'] !=0:
            for i in range(gt_p.shape[0]):
                if i in fn_gt_index:                
                    cv2.circle(img,(gt_p[i][0],gt_p[i][1]),point_r_value,(0,0,255),-1)# fn: red
                    cv2.circle(img,(gt_p[i][0],gt_p[i][1]),sigma_l[i],(0,0,255),thickness)#  
                else:
                    cv2.circle(img,(gt_p[i][0],gt_p[i][1]),sigma_l[i],(0,255,0),thickness)# gt: green
        if pred_data[i_sample]['num'] !=0:
            for i in range(pred_p.shape[0]):
                if i in tp_pred_index:
                    cv2.circle(img,(pred_p[i][0],pred_p[i][1]),point_r_value,(0,255,0),-1)# tp: green
                else:                
                    cv2.circle(img,(pred_p[i][0],pred_p[i][1]),point_r_value*2,(255,0,255),-1) # fp: Magenta

        # cv2.imwrite(exp_name+'/'+str(i_sample)+ '_pre_' + str(pre)[0:6] + '_rec_' + str(rec)[0:6] + '.jpg', img)
        output_file = exp_name / f"{i_sample:04d}_pre_{str(pre)[:6]}_rec_{str(rec)[:6]}.jpg"
        cv2.imwrite(str(output_file), img)




if __name__ == '__main__':
    main()
