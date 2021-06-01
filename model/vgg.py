import torch
import torch.nn as nn

__all__ = ['VGG']
class VGG(nn.Module):
    
    def __init__(self,num_input=3,num_label=10):
        super(VGG,self).__init__()
        self.channel_dict = {
            'conv_cls_1':[64,64],
            'conv_cls_2':[128,128],
            'conv_cls_3':[256,256,256],
            'conv_cls_4':[512,512,512],
            'conv_cls_5':[512,512,512],
            'fc_6':[4096,4096,num_label],
        }

        self.conv_cls_1=nn.Sequential(
            nn.Conv2d(in_channels=num_input,out_channels=self.channel_dict['conv_cls_1'][0],
                      kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(in_channels=self.channel_dict['conv_cls_1'][0],out_channels=self.channel_dict['conv_cls_1'][1],
                      kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),           
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.conv_cls_2=nn.Sequential(
            nn.Conv2d(in_channels=self.channel_dict['conv_cls_1'][1],out_channels=self.channel_dict['conv_cls_2'][0],
                      kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.channel_dict['conv_cls_2'][0],out_channels=self.channel_dict['conv_cls_2'][1],
                      kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.conv_cls_3=nn.Sequential(
            nn.Conv2d(in_channels=self.channel_dict['conv_cls_2'][1],out_channels=self.channel_dict['conv_cls_3'][0],
                      kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=self.channel_dict['conv_cls_3'][0],out_channels=self.channel_dict['conv_cls_3'][1],
                      kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.channel_dict['conv_cls_3'][1],out_channels=self.channel_dict['conv_cls_3'][2],
                      kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.conv_cls_4=nn.Sequential(
            nn.Conv2d(in_channels=self.channel_dict['conv_cls_3'][2],out_channels=self.channel_dict['conv_cls_4'][0],
                      kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.channel_dict['conv_cls_4'][0],out_channels=self.channel_dict['conv_cls_4'][1],
                      kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.channel_dict['conv_cls_4'][1],out_channels=self.channel_dict['conv_cls_4'][2],
                      kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.conv_cls_5=nn.Sequential(
            nn.Conv2d(in_channels=self.channel_dict['conv_cls_4'][2],out_channels=self.channel_dict['conv_cls_5'][0],
                      kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.channel_dict['conv_cls_5'][0],out_channels=self.channel_dict['conv_cls_5'][1],
                      kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.channel_dict['conv_cls_5'][1],out_channels=self.channel_dict['conv_cls_5'][2],
                      kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.AvgPool2d(kernel_size=1,stride=1),
        )

        self.fc_6 =  nn.Sequential(
             nn.Linear(in_features=self.channel_dict['conv_cls_5'][2]*1*1,out_features=self.channel_dict['fc_6'][0]),
             nn.ReLU(inplace=True),
             nn.Dropout(),
             nn.Linear(in_features=self.channel_dict['fc_6'][0],out_features=self.channel_dict['fc_6'][1]),
             nn.ReLU(inplace=True),   
             nn.Dropout(),
             nn.Linear(in_features=self.channel_dict['fc_6'][1],out_features=self.channel_dict['fc_6'][2]),
                )

    def forward(self,x):
        cls_feature_1 = self.conv_cls_1(x)
        # print('cls_feature_1 size is:{}'.format(cls_feature_1.size()))
        cls_feature_2 = self.conv_cls_2(cls_feature_1)
        # print('cls_feature_2 size is:{}'.format(cls_feature_2.size()))

        cls_feature_3 = self.conv_cls_3(cls_feature_2)
        # print('cls_feature_3 size is:{}'.format(cls_feature_3.size()))

        cls_feature_4 = self.conv_cls_4(cls_feature_3)
        # print('cls_feature_4 size is:{}'.format(cls_feature_4.size()))

        cls_feature_5 = self.conv_cls_5(cls_feature_4)
        # print('cls_feature_5 size is:{}'.format(cls_feature_5.size()))

        conv_feature = cls_feature_5.view(cls_feature_5.size(0),-1)
        # print('conv_feature size is:{}'.format(conv_feature.size()))

        output_feature = self.fc_6(conv_feature)

        return output_feature

#check model
# def main():
#     network = VGG()
#     test_input = torch.Tensor(1,3,224,224)
#     output = network(test_input)
#     print(network)
#     print(output.size())

# if __name__ =="__main__":
#     main()
    


        

