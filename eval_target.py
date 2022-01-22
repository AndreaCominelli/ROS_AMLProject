
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
        for it, (data,class_l,data_rot,rot_l) in tqdm(enumerate(target_loader_eval)):
            data, class_l, data_rot, rot_l = data.to(device), class_l.to(device), data_rot.to(device), rot_l.to(device)

            imgs_out = feature_extractor(data)
            rot_out = feature_extractor(data_rot)
            rot_predictions = rot_cls(torch.cat((rot_out, imgs_out), dim=1))

            normality_score, _ = torch.max(rot_predictions, 1)
            normality_scores.append(normality_score)
            ground_truth.append(rot_l)

    print(normality_scores)
    print(normality_scores)

    normality_scores = np.ndarray(normality_scores)
    ground_truth = np.ndarray(ground_truth)

    normality_scores = normality_scores.flatten()
    ground_truth = ground_truth.flatten()
    
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






