
import torch
from torch import nn
from optimizer_helper import get_optim_and_scheduler
from tqdm import tqdm

#### Implement Step1

def _do_epoch(args,feature_extractor,rot_cls,obj_cls,dataloaders,optimizer,device,phase,criterion):
    
    if phase == "train":
        print("TRAINING MODE")
        feature_extractor.train()
        obj_cls.train()
        rot_cls.train()
    else:
        print("VALIDATION MODE")
        feature_extractor.eval()
        obj_cls.eval()
        rot_cls.eval()

    #running_loss = 0
    running_img_corrects = 0
    running_rot_corrects = 0

    for i, (imgs, lbls, rot_imgs, rot_lbls) in tqdm(enumerate(dataloaders[phase])):
        imgs = imgs.to(device)
        lbls = lbls.to(device)
        rot_imgs = rot_imgs.to(device)
        rot_lbls = rot_lbls.to(device)

        # forward
        with torch.set_grad_enabled(phase == "train"):
            imgs_out = feature_extractor(imgs)
            imgs_predictions = obj_cls(imgs_out)
            
            rot_out = feature_extractor(rot_imgs)
            rot_predictions = rot_cls(rot_out)
            
            _, imgs_preds = torch.max(imgs_predictions, 1)
            _, rot_preds = torch.max(rot_predictions, 1)

            # compute loss

            img_loss = criterion(imgs_predictions, lbls)
            rot_loss = criterion(rot_predictions, rot_lbls)

            loss = img_loss + args.weight_RotTask_step1*rot_loss

            if phase == "train":
                    # compute gradient + update params
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # statistics
            # raccolgo in ciascun batch la loss per quel batch e quanti elementi sono stati
            # classificati correttamente

        #running_loss += loss.item() * imgs.size(0)
        running_img_corrects += torch.sum(imgs_preds == lbls.data)
        running_rot_corrects += torch.sum(rot_preds == rot_lbls.data)

    #class_loss = running_loss / dataset_sizes[phase]
    img_acc = (running_img_corrects.double() / len(dataloaders[phase].dataset)) * 100
    rot_acc = (running_rot_corrects.double() / len(dataloaders[phase].dataset)) * 100

    return img_loss, img_acc, rot_loss, rot_acc


def step1(args,feature_extractor,rot_cls,obj_cls,source_loader,device):
    optimizer, scheduler = get_optim_and_scheduler(feature_extractor,rot_cls,obj_cls, args.epochs_step1, args.learning_rate, args.train_all)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs_step1):
        print('Epoch: ',epoch)
        for phase in ["train", "val"]:
            class_loss, acc_cls, rot_loss, acc_rot = _do_epoch(args,feature_extractor,rot_cls,obj_cls,source_loader,optimizer,device, phase,criterion)
            print("Class Loss %.4f, Class Accuracy %.4f, Rot Loss %.4f, Rot Accuracy %.4f" % (class_loss.item(),acc_cls,rot_loss.item(), acc_rot))
        
            if phase == "train":    
                scheduler.step()
