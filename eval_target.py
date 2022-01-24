
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import random
from tqdm import tqdm


#### Implement the evaluation on the target for the known/unknown separation

def evaluation(args,feature_extractor,rot_cls,target_loader_eval,device):

    feature_extractor.eval()
    rot_cls.eval()
    
    normality_scores = []
    ground_truth = []

    with torch.no_grad():
        for it, (img ,class_l, img_90, img_180, img_270) in tqdm(enumerate(target_loader_eval)):
            img ,class_l, img_90, img_180, img_270 = img.to(device), class_l.to(device), img_90.to(device), img_180.to(device), img_270.to(device)
            
            if class_l > args.n_classes_known:
                ground_truth.append(0)
            else:
                ground_truth.append(1)

            rot_out_0   = feature_extractor(img)
            rot_out_90  = feature_extractor(img_90)
            rot_out_180 = feature_extractor(img_180)
            rot_out_270 = feature_extractor(img_270)

            rot_predictions_0   = rot_cls(torch.cat((rot_out_0, rot_out_0), dim=1))
            rot_predictions_90  = rot_cls(torch.cat((rot_out_90, rot_out_0), dim=1))
            rot_predictions_180 = rot_cls(torch.cat((rot_out_180, rot_out_0), dim=1))
            rot_predictions_270 = rot_cls(torch.cat((rot_out_270, rot_out_0), dim=1))
            

            normality_score_0, _   = torch.max(rot_predictions_0, 1)
            normality_score_90, _  = torch.max(rot_predictions_90, 1)
            normality_score_180, _ = torch.max(rot_predictions_180, 1)
            normality_score_270, _ = torch.max(rot_predictions_270, 1)

            normality_score = np.mean([normality_score_0.item(), normality_score_90.item(), normality_score_180.item(), normality_score_270.item()])[0]

            normality_scores.append(normality_score)

    """print(normality_scores)
    print(ground_truth)"""

    """normality_scores = np.ndarray(normality_scores)
    ground_truth = np.ndarray(ground_truth)

    normality_scores = normality_scores.flatten()
    ground_truth = ground_truth.flatten()"""
    
    auroc = roc_auc_score(ground_truth, normality_scores)
    print('AUROC %.4f' % auroc)

    # create new txt files
    """rand = random.randint(0,100000)
    print('Generated random number is :', rand)

    # This txt files will have the names of the source images and the names of the target images selected as unknown
    target_unknown = open('new_txt_list/' + args.source + '_known_' + str(rand) + '.txt','w')

    # This txt files will have the names of the target images selected as known
    target_known = open('new_txt_list/' + args.target + '_known_' + str(rand) + '.txt','w')"""



    """print('The number of target samples selected as known is: ',number_of_known_samples)
    print('The number of target samples selected as unknown is: ', number_of_unknown_samples)"""

    #return rand






