from PF2nlfff.packages import *
from PF2nlfff.mf_module import *
from PF2nlfff.Loss_function import *
from PF2nlfff.tools import *

# ========================================================== #
#             U-Net generator                  #
# ========================================================== #
class DoubleConvolution(nn.Module):
    """
    ### Two $3 \times 3$ Convolution Layers

    Each step in the contraction path and expansive path have two $3 \times 3$
    convolutional layers followed by ReLU activations.

    In the U-Net paper they used $0$ padding,
    but we use $1$ padding so that final feature map is not cropped.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        :param in_channels: is the number of input channels
        :param out_channels: is the number of output channels
        """
        super().__init__()

        # First $3 \times 3$ convolutional layer
        self.first = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU()
        # Second $3 \times 3$ convolutional layer
        self.second = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU()

    def forward(self, x: torch.Tensor):
        # Apply the two convolution layers and activations
        x = self.first(x)
#         x = self.bn1(x)
        x = self.act1(x)
        x = self.second(x)
#         x = self.bn2(x)
        return self.act2(x)


class DownSample(nn.Module):
    """
    ### Down-sample

    Each step in the contracting path down-samples the feature map with
    a $2 \times 2$ max pooling layer.
    """

    def __init__(self):
        super().__init__()
        # Max pooling layer
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        return self.pool(x)


class UpSample(nn.Module):
    """
    ### Up-sample

    Each step in the expansive path up-samples the feature map with
    a $2 \times 2$ up-convolution.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # Up-convolution
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        return self.up(x)


class CropAndConcat(nn.Module):
    """
    ### Crop and Concatenate the feature map

    At every step in the expansive path the corresponding feature map from the contracting path
    concatenated with the current feature map.
    """
    def forward(self, x: torch.Tensor, contracting_x: torch.Tensor):
        """
        :param x: current feature map in the expansive path
        :param contracting_x: corresponding feature map from the contracting path
        """

        # Crop the feature map from the contracting path to the size of the current feature map
        contracting_x = torchvision.transforms.functional.center_crop(contracting_x, [x.shape[2], x.shape[3]])
        # Concatenate the feature maps
        x = torch.cat([x, contracting_x], dim=1)
        #
        return x


class UNet(nn.Module):
    """
    ## U-Net
    """
    def __init__(self, in_channels: int, out_channels: int):
        """
        :param in_channels: number of channels in the input image
        :param out_channels: number of channels in the result feature map
        """
        super().__init__()

        # Double convolution layers for the contracting path.
        # The number of features gets doubled at each step starting from $64$.
        self.down_conv = nn.ModuleList([DoubleConvolution(i, o) for i, o in
                                        [(in_channels, 64), (64, 128), (128, 256), (256, 512)]])
        # Down sampling layers for the contracting path
        self.down_sample = nn.ModuleList([DownSample() for _ in range(4)])

        # The two convolution layers at the lowest resolution (the bottom of the U).
        self.middle_conv = DoubleConvolution(512, 1024)

        # Up sampling layers for the expansive path.
        # The number of features is halved with up-sampling.
        self.up_sample = nn.ModuleList([UpSample(i, o) for i, o in
                                        [(1024, 512), (512, 256), (256, 128), (128, 64)]])
        # Double convolution layers for the expansive path.
        # Their input is the concatenation of the current feature map and the feature map from the
        # contracting path. Therefore, the number of input features is double the number of features
        # from up-sampling.
        self.up_conv = nn.ModuleList([DoubleConvolution(i, o) for i, o in
                                      [(1024, 512), (512, 256), (256, 128), (128, 64)]])
        # Crop and concatenate layers for the expansive path.
        self.concat = nn.ModuleList([CropAndConcat() for _ in range(4)])
        # Final $1 \times 1$ convolution layer to produce the output
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
        # Xavier初始化
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor):
        """
        :param x: input image
        """
        # To collect the outputs of contracting path for later concatenation with the expansive path.
        pass_through = []
        # Contracting path
        for i in range(len(self.down_conv)):
            # Two $3 \times 3$ convolutional layers
            x = self.down_conv[i](x)
            # Collect the output
            pass_through.append(x)
            # Down-sample
            x = self.down_sample[i](x)

        # Two $3 \times 3$ convolutional layers at the bottom of the U-Net
        x = self.middle_conv(x)

        # Expansive path
        for i in range(len(self.up_conv)):
            # Up-sample
            x = self.up_sample[i](x)
            # Concatenate the output of the contracting path
            x = self.concat[i](x, pass_through.pop())
            # Two $3 \times 3$ convolutional layers
            x = self.up_conv[i](x)

        # Final $1 \times 1$ convolution layer
        x = self.final_conv(x)

        #
        return x
    
# ============================================================== #
#                 Discriminator                  #
# ============================================================== #
class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
#         if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
#             use_bias = norm_layer.func == nn.InstanceNorm2d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm2d
        use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)
    
# ==================================================== #
#      generated adversal network              #
# ==================================================== #
class GAN_extrapolation():
    def __init__(self, G_in, G_out, ub, sample_size, batch_size, PF_data, nlfff_data):
        
        self.G_in = G_in
        self.G_out = G_out
        self.ub = ub
        self.size = sample_size
        self.batch_size = batch_size
        self.nlfff_data = nlfff_data
        self.PF_data = PF_data
        
        # CUDA support 
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.netG = UNet(G_in, G_out).to(self.device)
        self.netD = NLayerDiscriminator(G_in+G_out).to(self.device)
        
        self.loss_GAN = GANLoss('vanilla').to(self.device)
        self.loss_L1 = torch.nn.L1Loss().to(self.device)
        
        self.iter = 0
        self.Gloss_list = []
        self.Dloss_list = []
        
        self.L1_lambda = 100.
        
        self.print_epoch = 100
        self.save_epoch = 1000
        self.plot_epoch = 1000
        
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=1.e-3, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=1.e-3, betas=(0.5, 0.999))

        self.scheduler_G = StepLR(self.optimizer_G, step_size=1000, gamma=0.9)
        self.scheduler_D = StepLR(self.optimizer_D, step_size=1000, gamma=0.9)
        
    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake
        self.fake_AB = torch.cat((self.real_A,self.fake_B), 1)
        pred_fake = self.netD(self.fake_AB.detach())
        self.loss_D_fake = self.loss_GAN(pred_fake, False)
        # Real
        self.real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(self.real_AB)
        self.loss_D_real = self.loss_GAN(pred_real, True)
        # combine the loss and calculate gradients
        self.loss_D = (self.loss_D_fake+self.loss_D_real)*0.5
#         print('loss_D: %.5e' % self.loss_D.item()) # test
        self.loss_D.backward()
        self.Dloss_list.append(self.loss_D.item())
        
    def backward_G(self):
        """Calculate GAN and L1 Loss for the generator"""
        # Fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.loss_GAN(pred_fake, True)
        # L1 loss :-> G(A)==B
        self.loss_G_L1 = self.loss_L1(self.fake_B, self.real_B)*self.L1_lambda
        # combine the loss and calculate the gradients
        self.loss_G = self.loss_G_GAN+self.loss_G_L1
        self.loss_G.backward()
        self.Gloss_list.append(self.loss_G.item())
        
    def latent_sample(self):
        latent = []
        sample = []
        bottom_boundary = self.nlfff_data[:,:,:,0]
        for i in range(self.batch_size):
            x_ind = np.random.randint(0, self.ub[0]-self.size)
            y_ind = np.random.randint(0, self.ub[1]-self.size)
            z_ind = np.random.randint(1, self.ub[2])
            
            layeri = self.PF_data[:,x_ind:x_ind+self.size,y_ind:y_ind+self.size,z_ind]
            layer0 = bottom_boundary[:,x_ind:x_ind+self.size,y_ind:y_ind+self.size]
            b0_max = np.abs(layer0).max(axis=(1,2))
            bi_max = np.abs(layeri).max(axis=(1,2))
            wi = bi_max/b0_max
            
            latent.append(
                np.vstack([layeri, layer0*wi[:,np.newaxis,np.newaxis]]))
            sample.append(self.nlfff_data[:,x_ind:x_ind+self.size,y_ind:y_ind+self.size,z_ind])
#             sample.append(
#                 nlfff_data[:,x_ind:x_ind+self.size,y_ind:y_ind+self.size,z_ind]/
#                 layer_std[:,np.newaxis, np.newaxis])

        latent = torch.tensor(np.array(latent)).float().to(self.device)
        sample = torch.tensor(np.array(sample)).float().to(self.device)

        return latent, sample

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
    
    def train(self, epoch=100, lr=1e-3, beta1=0.5, Delta=500, gamma=1./np.exp(1), is_plot = False):
        
        if self.iter ==0:
            self.wall_time = 0.
        start_time = time.time()
        if lr == 'continue':
            first = True
        else:
            first = True
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = lr
            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = lr
        self.scheduler_G.gamma = gamma
        self.scheduler_G.step_size = Delta
        self.scheduler_D.gamma = gamma
        self.scheduler_D.step_size = Delta        
        
        for iepoch in range(epoch):
            self.real_A, self.real_B = self.latent_sample()
            self.fake_B = self.netG(self.real_A)

            # update D
            self.set_requires_grad(self.netD, True)  # enable backprop for D
            self.optimizer_D.zero_grad()     # set D's gradients to zero
            self.backward_D()                # calculate gradients for D
            self.optimizer_D.step()          # update D's weights
            self.scheduler_D.step()
            # update G
            self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
            self.optimizer_G.zero_grad()        # set G's gradients to zero
            self.backward_G()                   # calculate graidents for G
            self.optimizer_G.step()             # update G's weights
            self.scheduler_G.step()

            if self.iter % self.print_epoch ==0:
                time_now = time.time()
                time_used = time_now-start_time+self.wall_time
                print(
                'Iter %05d, Loss_D: %.4e, Loss_G: %.4e, wall_time: %.4e sec, lr: %.4e' % \
                    (self.iter, self.Dloss_list[-1], self.Gloss_list[-1], time_used, 
                     self.scheduler_G.get_last_lr()[-1])
                )
            
            if not os.path.exists( './trained_model/models/'):
                os.makedirs( './trained_model/models/')
                print('create the DIR. : ./trained_model/models/')            
            if self.iter % self.save_epoch == 0:
                torch.save(self, './trained_model/models/PF2nlfff_%05d.pkl' % self.iter)
            if self.Gloss_list[-1]<=np.min(self.Gloss_list[:]):
                torch.save(self, './trained_model/models/best_model.pkl')
            
            # plotting
            if self.iter % self.plot_epoch ==0 and is_plot and self.iter>=self.plot_epoch:
                time_now = time.time()
                time_used = time_now-start_time+self.wall_time
                clear_output(wait=True)
                plt.figure(figsize=(10, 10))
                ax1 = plt.subplot(211)
                ax2 = plt.subplot(212, sharex=ax1)

                ax1.plot(self.Gloss_list[-self.plot_epoch:])
                ax1.set_yscale('log')
                ax1.set_ylabel('G_loss')
                ax1.set_title('Iter %05d, Loss_D: %.4e, Loss_G: %.4e, wall_time: %.4e sec, lr: %.4e' % \
                              (self.iter, self.Dloss_list[-1], self.Gloss_list[-1], time_used, 
                               self.scheduler_G.get_last_lr()[-1]))

                ax2.plot(self.Dloss_list[-self.plot_epoch:])
                ax2.set_yscale('log')
                ax2.set_ylabel('D_loss')
                ax2.set_xlabel('Rounds in the last %d epochs' % self.plot_epoch)
                if self.iter % 100==0:
                    if not os.path.exists('./trained_model/training_pics/'):
                        os.makedirs( './trained_model/training_pics/')
                        print('create the DIR. : /trained_model/training_pics/')  
                    plt.savefig('./trained_model/training_pics/loss_%05d.png' % self.iter,
                                dpi=200, bbox_inches ="tight")
                plt.draw()
                plt.pause(0.01)
            
            self.iter+=1
            
        end_time = time.time()
        time_used = end_time-start_time
        self.wall_time += time_used
        
class trainer_network():
    def __init__(self, net, PF_data):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = net.to(self.device)
        self.PF_data = PF_data
        
        self.loss_func = nn.MSELoss().to(self.device)
#         self.loss_func = nn.L1Loss().to(self.device)
        
        self.iter = 0
        self.loss_list = []
        
        self.L_div = 1.e4
        self.L_ff = 1.e-1
        self.L_bc = 1.e0
        self.L_sine = 1.e1
        self.L_fi = 1.e4
        
        self.print_epoch = 10
        self.save_epoch = 100
        self.plot_epoch = 100
        self.auto_weights = False
        
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1.e-3, betas = (0.5, 0.999))
        self.scheduler = StepLR(self.optimizer, step_size=100, gamma=0.9)
        
        self.save_model_dir = './trainer/models'
        self.save_pic_dir =  './trainer/pics'
        
    def network_input(self):
        """Build the input of the network"""
        n1,n2,n3 = self.PF_data.shape[1:]
        ret = []
        layer0 = self.PF_data[:,:,:,0]
        for i in range(n3):
            layeri = self.PF_data[:,:,:,i]
            layeri_max = np.abs(layeri).max(axis = (1,2))
            layer0_max = np.abs(layer0).max(axis = (1,2))
            wi = layeri_max/layer0_max
            ret.append(np.vstack([layeri, layer0*wi[:,np.newaxis,np.newaxis]]))
        ret = np.array(ret)
        ret = torch.tensor(ret).float().to(self.device)
        return ret
    
    def Loss(self):
        """Calculate the divb and sigma_J with the finity difference"""
        """Calculate the loss for the boundary condition"""
        PF_in = self.network_input()
        b_cube = self.net(PF_in).permute(1,2,3,0)
        # boundary condition
        bottom_boundary = torch.tensor(self.PF_data[:,:,:,0:1]).float().to(self.device)
        boundary_out = self.net(PF_in).permute(1,2,3,0)[:,:,:,0:1]
        ff, div, self.sigma_J, self.fi = EQLoss(b_cube)
        
        if self.auto_weights:
            c1 = np.floor(np.log10(div))
            c2 = np.floor(np.log10(ff))
            c3 = np.floor(np.log10(self.loss_func(boundary_out, bottom_boundary).item()))
            self.L_div = 1/10**c1
            self.L_ff = 1/10**c2
            self.L_bc = 1/10**c3

        # loss
#         loss_div = self.L_div*self.loss_func(div,torch.zeros_like(div))
#         loss_ff = self.L_ff*self.loss_func(ff_factor,torch.zeros_like(ff_factor))+self.L_sine*self.sigma_J
        loss_div = self.L_div*div+self.L_fi*self.f_i
        loss_ff = self.L_ff*ff + self.L_sine*self.sigma_J
        loss_bc = self.L_bc*self.loss_func(boundary_out, bottom_boundary)
        return loss_div, loss_ff, loss_bc
        

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
    
    def train(self, epoch=100, lr=1e-3, Delta=500, gamma=1./np.exp(1), is_plot = False):
        
        if self.iter ==0:
            self.wall_time = 0.
        start_time = time.time()
        
        self.set_requires_grad(self.net, True)
        if lr == 'continue':
            first = True
        else:
            first = True
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        self.scheduler.gamma = gamma
        self.scheduler.step_size = Delta
        
        for iepoch in range(epoch):
            # update network
            self.optimizer.zero_grad()
            self.loss_div, self.loss_ff, self.loss_bc = self.Loss()
            self.loss = self.loss_div+self.loss_ff+self.loss_bc
            self.loss.backward()
            self.loss_list.append(self.loss.item())
            self.optimizer.step()
            self.scheduler.step()            

            # print training information
            if self.iter % self.print_epoch ==0 or first:
                time_now = time.time()
                time_used = time_now-start_time+self.wall_time
                print(
                'Iter %05d, Loss: %.3e, Loss_div: %.3e,Loss_ff: %.3e, Loss_bc: %.3e, wall_time: %.3e sec, lr: %.3e' % \
                    (self.iter, self.loss_list[-1], self.loss_div, self.loss_ff, self.loss_bc, time_used, 
                     self.scheduler.get_last_lr()[-1])
                )
            first = False
            
            if self.iter % self.save_epoch ==0:
                if not os.path.exists( self.save_model_dir):
                    os.makedirs( self.save_model_dir)
                    print('create the DIR. : %s' % self.save_model_dir)
                torch.save(self, self.save_model_dir+'/PF2nlfff_with_trainer_%05d.pkl' % self.iter)
            
            # plotting
            if self.iter % self.plot_epoch ==0 and self.iter>=self.plot_epoch and is_plot:
                time_now = time.time()
                time_used = time_now-start_time+self.wall_time
                clear_output(wait=True)
                plt.figure(figsize=(10, 6.18))
                ax1 = plt.subplot(111)

                ax1.plot(self.loss_list[-self.plot_epoch:])
                ax1.set_yscale('log')
                ax1.set_ylabel('loss')
                ax1.set_title(
                    'Iter %05d, Loss: %.3e, Loss_div: %.3e,Loss_ff: %.3e, Loss_bc: %.3e, time: %.3e s, lr: %.3e' % \
                    (self.iter, self.loss_list[-1], self.loss_div, self.loss_ff, self.loss_bc, time_used, 
                     self.scheduler.get_last_lr()[-1])
                )

                if self.iter % self.plot_epoch==0:
                    if not os.path.exists( self.save_pic_dir):
                        os.makedirs( self.save_pic_dir)
                        print('create the DIR. : %s' % self.save_pic_dir)
                    plt.savefig(self.save_pic_dir+'/loss_%05d.png' % self.iter,
                                dpi=200, bbox_inches ="tight")
                plt.draw()
                plt.pause(0.01)
            
            self.iter+=1
            
        end_time = time.time()
        time_used = end_time-start_time
        self.wall_time += time_used
