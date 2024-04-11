import torch
import torch.nn as nn

class UNet(nn.Module):

    def __init__(self, n_class):

        super().__init__()
        self.n_class = n_class
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv9 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv10 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, dilation=1)
        
        self.relu = nn.ReLU(inplace=True)
        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv11 = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv12 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv13 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv14 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        
        self.deconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, padding=0, dilation=1, output_padding=0)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0, dilation=1, output_padding=0)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0, dilation=1, output_padding=0)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0, dilation=1, output_padding=0)
        self.classifier = nn.Conv2d(64, self.n_class, kernel_size=1)
        
        self.bnd1 = nn.BatchNorm2d(64)
        self.bnd2 = nn.BatchNorm2d(128)
        self.bnd3 = nn.BatchNorm2d(256)
        self.bnd4 = nn.BatchNorm2d(512)
        self.bnd5 = nn.BatchNorm2d(1024)
        
        self.dropout = nn.Dropout(p=0.3) # Anywhere between 0.2 and 0.5 is valid


    def forward(self, x):
        # Start of contraction
        
        x1 = self.relu(self.bnd1(self.conv1(x))) # n_channel = 64 (N,64,224,224)
        #print(x1.shape)
        x2 = self.relu(self.bnd1(self.conv2(x1))) # n_channel = 64 (N,64,224,224)
        #print(x2.shape)
        p1 = self.maxpool2d(x2) # n_channel = 64 (N,64,112,112)
        #print(p1.shape)
        
        x3 = self.relu(self.bnd2(self.conv3(p1))) # n_channel = 128 (N,128,112,112)
        #print(x3.shape)
        x4 = self.relu(self.bnd2(self.conv4(x3))) # n_channel = 128 (N,128,112,112)
        #print(x4.shape)
        p2 = self.maxpool2d(x4) # n_channel = 128 (N,128,56,56)
        #print(p2.shape)
        
        x5 = self.relu(self.bnd3(self.conv5(p2))) # n_channel = 256 (N,256,56,56)
        #print(x5.shape)
        x6 = self.relu(self.bnd3(self.conv6(x5))) # n_channel = 256 (N,256,56,56)
        #print(x6.shape)
        p3 = self.maxpool2d(x6) # n_channel = 256 (N,256,28,28)
        #print(p3.shape)
        
        x7 = self.relu(self.bnd4(self.conv7(p3))) # n_channel = 512 (N,512,28,28)
        #print(x7.shape)
        x8 = self.relu(self.bnd4(self.conv8(x7))) # n_channel = 512 (N,512,28,28)
        #print(x8.shape)
        p4 = self.maxpool2d(x8) # n_channel = 512 (N,512,14,14)
        #print(p4.shape)
        
        x9 = self.relu(self.bnd5(self.conv9(p4))) # n_channel = 1024 (N,1024,14,14)
        #print(x9.shape)
        x10 = self.dropout(self.relu(self.bnd5(self.conv10(x9)))) # n_channel = 1024 (N,1024,14,14)
        #print(x10.shape)
         
        # End of contraction
        
        # Start of Expansion
        y1 = self.bnd4(self.deconv1(x10)) # n_channel = 512 (N,512,28,28)
        #print(y1.shape) 
        y2 = self.relu(self.bnd4(self.conv11(torch.cat((x8, y1), dim=1)))) # n_channel = 512 (N,512,28,28)
        #print(y2.shape)
        y3 = self.relu(self.bnd4(self.conv8(y2))) # n_channel = 512 (N,512,28,28)
        #print(y3.shape)
        
        y4 = self.bnd3(self.deconv2(y3)) # n_channel = 256 (256,56,56)
        #print(y4.shape)
        y5 = self.relu(self.bnd3(self.conv12(torch.cat((x6, y4), dim=1)))) # n_channel = 256 (N,256,56,56)
        #print(y5.shape)
        y6 = self.relu(self.bnd3(self.conv6(y5))) # n_channel = 256 (N,256,56,56)
        #print(y6.shape)
        
        y7 = self.bnd2(self.deconv3(y6)) # n_channel = 128 (128,112,112)
        #print(y7.shape)
        y8 = self.relu(self.bnd2(self.conv13(torch.cat((x4, y7), dim=1)))) # n_channel = 128 (N,128,112,112)
        #print(y8.shape)
        y9 = self.relu(self.bnd2(self.conv4(y8))) # n_channel = 128 (N,128,112,112)
        #print(y9.shape)
        
        y10 = self.bnd1(self.deconv4(y9)) # n_channel = 64 (N,64,224,224)
        #print(y10.shape)
        y11 = self.relu(self.bnd1(self.conv14(torch.cat((x2, y10), dim=1)))) # n_channel = 64 (N,64,224,224)
        #print(y11.shape)
        y12 = self.relu(self.bnd1(self.conv2(y11))) # n_channel = 64 (N,64,224,224)
        #print(y12.shape)

        score = self.classifier(y12) # n_channel = 21 (N,21,224,224)
        
        # End of expansion 
        
        return score  # size=(N, n_class, H, W)