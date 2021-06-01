import os
import torch
import torch.nn as nn

__all__ = ["CheckPoint"]

class CheckPoint(object):

    def __init__(self,save_path):
        self.save_path = os.path.join(save_path,"check_point")
        # make direction
        if not os.path.isdir(self.save_path):
            print(">>> Path not exists ,create path {}".format(self.save_path))
            os.makedirs(self.save_path)
    def save_checkpoint(self,model,optimizer,epoch,name,extra_params=None):
        
        self.check_point_params = {'model': None,
                                   'optimizer': None,
                                   'epoch': None,
                                   'extra_params': None}
        model.eval()
                # get state_dict from model and optimizer
        if isinstance(model, nn.DataParallel):
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()

        optimizer_state_dict = optimizer.state_dict()
        # save information to a dict
        self.check_point_params['model'] = model_state_dict
        self.check_point_params['optimizer'] = optimizer_state_dict
        self.check_point_params['epoch'] = epoch + \
            1  # add 1 to start a new epoch
        self.check_point_params['extra_params'] = extra_params

        # save to file
        torch.save(self.check_point_params, os.path.join(
            self.save_path, name))
    def save_state_dict(self, model, name):
        '''save state dict of model

        Arguments:
            model {torch.nn.Module} -- model to save
            name {bool} -- name of saved file
        '''

        # get state dict
        if isinstance(model, nn.DataParallel):
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()
        torch.save(model_state_dict, os.path.join(self.save_path, name))

