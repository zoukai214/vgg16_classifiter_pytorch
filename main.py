
import datetime
import time
import yaml

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from trainer import Trainer
from model.vgg import VGG

from checkpoint.checkpoint import CheckPoint
class ExperimentMaster(object):
    def __init__(self):
        ''' init class:define a standard experiment pipeline
        
        '''
        file = open('options_local.yml')
        self.settings = yaml.load(file,Loader = yaml.FullLoader)
    
        self.save_new_path = "cifar10_vgg_cls"
        
        # initialize gpu
        torch.manual_seed(self.settings['run_config']['manual_seed'])
        torch.cuda.manual_seed(self.settings['run_config']['manual_seed'])
        torch.cuda.set_device(0)
        cudnn.benchmark = True
        
        #checkpoint for save models or optimizers
        self.checkpoint = CheckPoint(save_path= self.save_new_path)

        # self.checkpoint = 
        self.model = VGG(3,10)
        self.trainer = Trainer(settings=self.settings,model = self.model,save_path = self.save_new_path)
        print(">>> Init experiment master done!")

    def run(self):
        best_result_dict = None

        for epoch in range(0,self.settings['train_config']['epochs']):
            epoch_t_start  = time.time()
            #train
            self.trainer.run(epoch,mode="train")
            #testing
            self.trainer.run(epoch,mode="val")
        
            
    #         if result_dict is not None:
    #             if not isinstance(result_dict, dict):
    #                 raise TypeError("trainer should return a dict, while it returns {}".format(type(result_dict)))


    # #             # save best result
    # #             # if self.local_rank == 0:
    #             if best_result_dict is None:
    #                 best_result_dict = result_dict

    #             if result_dict is not None:
                    # for k, v in best_result_dict.items():
                    #     if result_dict[k] >= best_result_dict[k]:
                    #         try:
                    #             self.checkpoint.save_state_dict(
                    #                 model=self.model,
                    #                 name="{}.pth".format(k)
                    #             )
    #                         except Exception as msg:
    #                             print(msg.value)
    #                         print("Get best {}: {}".format(k, v))

    #             epoch_t = time.time()-epoch_t_start
    #             epoch_t_meter.update(epoch_t)
    #             print(">>> Avg epoch time: %s, Remaining time: %s" % (
    #                 datetime.timedelta(seconds=epoch_t_meter.avg),
    #                 datetime.timedelta(seconds=epoch_t_meter.avg*(self.settings['train_config']['epochs'])-epoch-1)))

                    # save checkpoint
                    
            self.checkpoint.save_checkpoint(
                model=self.model,
                optimizer=self.trainer.optimizer,
                epoch=epoch,
                name='checkpoint_vgg.pth')
                  
    #     try:
    #         self.checkpoint.save_state_dict(
    #             model=self.model,
    #             name='model.pth')
    #     except Exception as msg:
    #                 print(msg)



def main():
    print("start---------------------")
    exp = ExperimentMaster()
    exp.run()


if __name__ == '__main__':
    main()