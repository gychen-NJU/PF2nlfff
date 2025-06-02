from PF2nlfff.packages import *
from PF2nlfff.nlfff_module import *
from PF2nlfff.Loss_function import *
from PF2nlfff.tools import *

# ===================================================== #
#              Discriminator                            #
# ===================================================== #
class NLayerDiscriminator_3D(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input blocks
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        # super(NLayerDiscriminator_3D, self).__init__()
        super().__init__()
        use_bias = norm_layer == nn.InstanceNorm3d

        kw = 4
        padw = 1
        sequence = [nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)
    
# ===================================================== #
#                     3D-Unet                           #
# ===================================================== #        
class UNet3D(Module):
    # __                            __
    #  1|__   ________________   __|1
    #     2|__  ____________  __|2
    #        3|__  ______  __|3
    #           4|__ __ __|4
    # The convolution operations on either side are residual subject to 1*1 Convolution for channel homogeneity

    def __init__(self,num_channels=3,feat_channels=[64, 128, 256, 512, 1024], residual='conv',encoder_device:str='default',decoder_device:str='default'):

        #residual: Whether to add residual edge or not, not:None
        # super(UNet3D, self).__init__()
        super().__init__()
        
        self.encoder_device = (torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
                               if encoder_device=='default' else torch.device(encoder_device))
        self.decoder_device = (torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
                               if encoder_device=='default' else torch.device(decoder_device))

        # Encoder downsamplers
        self.pool1 = MaxPool3d((1,2,2)).to(self.encoder_device)
        self.pool2 = MaxPool3d((1,2,2)).to(self.encoder_device)
        self.pool3 = MaxPool3d((1,2,2)).to(self.encoder_device)
        self.pool4 = MaxPool3d((1,2,2)).to(self.encoder_device)

        # Encoder convolutions
        self.conv_blk1 = Conv3D_Block(num_channels, feat_channels[0], residual=residual).to(self.encoder_device)
        self.conv_blk2 = Conv3D_Block(feat_channels[0], feat_channels[1], residual=residual).to(self.encoder_device)
        self.conv_blk3 = Conv3D_Block(feat_channels[1], feat_channels[2], residual=residual).to(self.encoder_device)
        self.conv_blk4 = Conv3D_Block(feat_channels[2], feat_channels[3], residual=residual).to(self.encoder_device)
        self.conv_blk5 = Conv3D_Block(feat_channels[3], feat_channels[4], residual=residual).to(self.encoder_device)

        # Decoder convolutions
        self.dec_conv_blk4 = Conv3D_Block(2*feat_channels[3], feat_channels[3], residual=residual).to(self.decoder_device)
        self.dec_conv_blk3 = Conv3D_Block(2*feat_channels[2], feat_channels[2], residual=residual).to(self.decoder_device)
        self.dec_conv_blk2 = Conv3D_Block(2*feat_channels[1], feat_channels[1], residual=residual).to(self.decoder_device)
        self.dec_conv_blk1 = Conv3D_Block(2*feat_channels[0], feat_channels[0], residual=residual).to(self.decoder_device)

        # Decoder upsamplers
        self.deconv_blk4 = Deconv3D_Block(feat_channels[4], feat_channels[3]).to(self.decoder_device)
        self.deconv_blk3 = Deconv3D_Block(feat_channels[3], feat_channels[2]).to(self.decoder_device)
        self.deconv_blk2 = Deconv3D_Block(feat_channels[2], feat_channels[1]).to(self.decoder_device)
        self.deconv_blk1 = Deconv3D_Block(feat_channels[1], feat_channels[0]).to(self.decoder_device)

        # Final 1*1 Conv Segmentation map
        self.one_conv = Conv3d(feat_channels[0], num_channels, kernel_size=1, stride=1, padding=0, bias=True).to(self.decoder_device)

        # Activation function
        self.sigmoid = Sigmoid().to(self.decoder_device)


    def forward(self, x):

        # encoder
        x1 = self.conv_blk1(x.to(self.encoder_device))
        x_low1 = self.pool1(x1)
        x2 = self.conv_blk2(x_low1)
        x_low2 = self.pool2(x2)
        x3 = self.conv_blk3(x_low2)
        x_low3 = self.pool3(x3)
        x4 = self.conv_blk4(x_low3)
        x_low4 = self.pool4(x4)
        base = self.conv_blk5(x_low4)
        
        # move tensor to decoder device
        base = base.to(self.decoder_device)
        x4   = x4.to(self.decoder_device)
        x3   = x3.to(self.decoder_device)
        x2   = x2.to(self.decoder_device)
        x1   = x1.to(self.decoder_device)


        # decoder
        d4 = torch.cat([self.deconv_blk4(base), x4], dim=1)
        d_high4 = self.dec_conv_blk4(d4)
        d3 = torch.cat([self.deconv_blk3(d_high4), x3], dim=1)
        d_high3 = self.dec_conv_blk3(d3)
        d2 = torch.cat([self.deconv_blk2(d_high3), x2], dim=1)
        d_high2 = self.dec_conv_blk2(d2)
        d1 = torch.cat([self.deconv_blk1(d_high2), x1], dim=1)
        d_high1 = self.dec_conv_blk1(d1)
        seg = self.one_conv(d_high1)

        return seg


class Conv3D_Block(Module):

    def __init__(self, inp_feat, out_feat, kernel=3, stride=1, padding=1, residual=None):

        super(Conv3D_Block, self).__init__()

        self.conv1 = Sequential(
                        Conv3d(inp_feat, out_feat, kernel_size=kernel,
                                    stride=stride, padding=padding, bias=True),
                        BatchNorm3d(out_feat),
                        ReLU())

        self.conv2 = Sequential(
                        Conv3d(out_feat, out_feat, kernel_size=kernel,
                                    stride=stride, padding=padding, bias=True),
                        BatchNorm3d(out_feat),
                        ReLU())

        self.residual = residual

        if self.residual is not None:
            self.residual_upsampler = Conv3d(inp_feat, out_feat, kernel_size=1, bias=False)

    def forward(self, x):

        res = x

        if not self.residual:
            return self.conv2(self.conv1(x))
        else:
            return self.conv2(self.conv1(x)) + self.residual_upsampler(res)


class Deconv3D_Block(Module):

    def __init__(self, inp_feat, out_feat, kernel=4, stride=2, padding=1):

        super(Deconv3D_Block, self).__init__()

        self.deconv = Sequential(
                        #3D反卷积
                        ConvTranspose3d(inp_feat, out_feat, kernel_size=(1,kernel,kernel),
                                    stride=(1,stride,stride), padding=(0, padding, padding), output_padding=0, bias=True),
                        ReLU())

    def forward(self, x):
        return self.deconv(x)


class ChannelPool3d(AvgPool1d):

    def __init__(self, kernel_size, stride, padding):

        super(ChannelPool3d, self).__init__(kernel_size, stride, padding)
        self.pool_1d = AvgPool1d(self.kernel_size, self.stride, self.padding, self.ceil_mode)

    def forward(self, inp):
        n, c, d, w, h = inp.size()
        inp = inp.view(n,c,d*w*h).permute(0,2,1)
        pooled = self.pool_1d(inp)
        c = int(c/self.kernel_size[0])
        return inp.view(n,c,d,w,h)
    

# ===================================================== #
#              GAN model for PF2nlfff                   #
# ===================================================== #
class GAN_model():
    def __init__(self, training_ds, batch_size,in_device = 'cpu',out_device = 'cpu',cut_size = 64):
        
        self.training_ds = training_ds
        self.batch_size = batch_size
        self.device = torch.device(out_device)
        self.in_device = in_device
        self.cut_size = cut_size
        self.is_cut = True
        self.file_name = None
        self.keep_file = False

        self.netG = UNet3D(encoder_device = self.in_device, decoder_device = self.device)
        self.netD = NLayerDiscriminator_3D(3+3).to(self.device)
        
        self.loss_GAN = GANLoss('vanilla').to(self.device)
        self.loss_L1 = torch.nn.L1Loss().to(self.device)
        
        self.iter = 0
        self.Gloss_list = []
        self.Dloss_list = []
        
        self.L1_lambda = 100.
        
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=1.e-3, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=1.e-3, betas=(0.5, 0.999))

        self.scheduler_G = StepLR(self.optimizer_G, step_size=100, gamma=0.9)
        self.scheduler_D = StepLR(self.optimizer_D, step_size=100, gamma=0.9)   
        
        self.save_epoch = 500
        self.print_epoch = 10
        self.plot_epoch = 500
        
        self.save_models_dir = './gan_model/3D_models/'
        self.save_pics_dir = './gan_model/3D_pics/'
        
    def backward_D(self, real_A, real_B, fake_B):
        """Calculate GAN loss for the discriminator"""
        # Fake
        fake_AB = torch.cat((real_A, fake_B), 1)
        pred_fake = self.netD(fake_AB.detach().to(self.device))
        self.loss_D_fake = self.loss_GAN(pred_fake, False)
        # Real
        real_AB = torch.cat((real_A, real_B), 1).to(self.device)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.loss_GAN(pred_real, True)
        # combine the loss and calculate gradients
        self.loss_D = (self.loss_D_fake+self.loss_D_real)*0.5
#         print('loss_D: %.5e' % self.loss_D.item()) # test
        self.loss_D.backward()
        self.Dloss_list.append(self.loss_D.item())
        
    def backward_G(self, real_A, real_B, fake_B):
        """Calculate GAN and L1 Loss for the generator"""
        # Fake the discriminator
        fake_AB = torch.cat((real_A, fake_B), 1).to(self.device)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.loss_GAN(pred_fake, True)
        # L1 loss :-> G(A)==B
        self.loss_G_L1 = self.loss_L1(fake_B, real_B)*self.L1_lambda
        # combine the loss and calculate the gradients
        self.loss_G = self.loss_G_GAN+self.loss_G_L1.to(self.device)
        self.loss_G.backward()
        self.Gloss_list.append(self.loss_G.item())

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
                    
    def generate_latent(self, nums=1, is_test=False, give_file:str=None):
        choice = np.random.randint(len(self.training_ds))
        file_names = self.training_ds[choice]
        file_idx = np.random.choice(len(file_names),(nums if len(file_names)>=nums else len(file_names)),replace=False)
        nlfff_data = []
        PF_data = []
        nlfff_boundary = []
        PF_boundary = []
        # for i,idx in enumerate(file_idx):
        for i in range(nums):
            ds_idx = np.random.randint(len(self.training_ds))
            filelist = self.training_ds[ds_idx]
            file_idx = np.random.randint(len(filelist))
            if give_file==None:
                nlfff_file = (filelist[file_idx] 
                              if (True if is_test else not(self.keep_file and self.file_name!=None)) 
                              else self.file_name)
            else:
                nlfff_file = give_file
            if not is_test:
                self.file_name = nlfff_file
            # nlfff_file = file_names[idx]
            PF_file = nlfff_file[:-8]+'PF_data.npy'
            nlfff = adc_shape(read_bcube(nlfff_file), is_print_info=False)
            potential = adc_shape(np.load(PF_file), is_print_info=False)
            n3 = np.min((nlfff.shape[-1],potential.shape[-1]))
            nlfff = nlfff[:,:,:,:n3]
            potential = potential[:,:,:,:n3]
            # nlfff_data.append(nlfff)
            # PF_data.append(potential)
            if self.is_cut:
                x_cut = np.random.randint(nlfff.shape[1]-self.cut_size-1)
                y_cut = np.random.randint(nlfff.shape[2]-self.cut_size-1)
                # z_cut = np.random.randint(nlfff.shape[3]-self.cut_size-1)
                z_cut = 0
                nlfff_data.append(nlfff[:,x_cut:x_cut+self.cut_size,y_cut:y_cut+self.cut_size,z_cut:z_cut+self.cut_size])
                PF_data.append(potential[:,x_cut:x_cut+self.cut_size,y_cut:y_cut+self.cut_size,z_cut:z_cut+self.cut_size])
            else:
                nlfff_data.append(nlfff)
                PF_data.append(potential)
        nlfff_data = np.array(nlfff_data)
        PF_data = np.array(PF_data)
        nlfff_tensor = torch.from_numpy(nlfff_data).float().to(self.in_device)
        PF_tensor = torch.from_numpy(PF_data).float().to(self.in_device)
        return nlfff_tensor, PF_tensor
    
    def train(self, epoch=100, lr=1e-3, Delta=500, gamma=0.9, is_plot = False, batch_epoch=100, **kwargs):
        ipt         = kwargs.get('ipt', None)
        opt         = kwargs.get('opt', None)
        save_best   = kwargs.get('save_best', False)
        save_first  = kwargs.get('save_first', False)
        save_path   = kwargs.get('save_path', self.save_models_dir)
        batch_size  = kwargs.get('batch_size', self.batch_size)
        if (ipt is None) or (opt is None):
            ipt,opt = self.return_io()
        total_batch = ipt.shape[0]
        self.save_models_dir = save_path
        
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
        self.scheduler_D.gamma = gamma
        self.scheduler_G.step_size = Delta
        self.scheduler_D.step_size = Delta
        
        print(f'### Start to train the GAN model [batch_size:total_batch={batch_size}/{total_batch}] ###')
        for iepoch in range(epoch):
            idx = np.random.choice(total_batch, total_batch, replace=False)
            ipt_new = ipt[idx]
            opt_new = opt[idx]
            nrounds = int(np.ceil(total_batch/batch_size))
            iloss_D = []
            iloss_G = []
            for iround in range(nrounds):
                hi = iround*batch_size
                ti = hi+batch_size
                real_A = torch.from_numpy(ipt_new[hi:ti]).float().to(self.device)
                real_B = torch.from_numpy(opt_new[hi:ti]).float().to(self.device)
                fake_B = self.netG(real_A)
                self.set_requires_grad(self.netD, True)
                # update D
                self.set_requires_grad(self.netD, True)  # enable backprop for D
                self.optimizer_D.zero_grad()     # set D's gradients to zero
                self.backward_D(real_A, real_B, fake_B)                # calculate gradients for D
                self.optimizer_D.step()          # update D's weights
                self.scheduler_D.step()
                # update G
                self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
                self.optimizer_G.zero_grad()        # set G's gradients to zero
                self.backward_G(real_A, real_B, fake_B)                   # calculate graidents for G
                self.optimizer_G.step()             # update G's weights
                self.scheduler_G.step()
                iloss_D.append(self.loss_D.item())
                iloss_G.append(self.loss_G.item())
                self.iter+=1
            self.Gloss_list.append(np.mean(iloss_G))
            self.Dloss_list.append(np.mean(iloss_D))
            # print training information
            if iepoch % self.print_epoch == 0 or first or iepoch+1==epoch:
                time_now = time.time()
                time_used = time_now-start_time+self.wall_time
                print(
                'Iter %05d, Loss_G: %.3e, Loss_D: %.3e, wall_time: %.3e sec, lr: %.3e' % \
                    (iepoch, self.Gloss_list[-1], self.Dloss_list[-1], time_used, 
                     self.scheduler_G.get_last_lr()[-1])
                )
            
            if iepoch % self.save_epoch == 0:
                if not os.path.exists(self.save_models_dir):
                    os.makedirs(self.save_models_dir, exist_ok=True)
                    print('create the DIR. : %s' % self.save_models_dir)
                if iepoch==0 and save_first:
                    torch.save(self, os.path.join(self.save_models_dir,'PF2nlfff_%05d.pkl' % iepoch))
                else:
                    torch.save(self, os.path.join(self.save_models_dir,'PF2nlfff_%05d.pkl' % iepoch))
            if len(self.Gloss_list)>1 and self.Gloss_list[-1]<=np.min(self.Gloss_list[:-1]) and self.iter >= 100:
                best_netG = copy.deepcopy(self.netG)
            first=False
            
        end_time = time.time()
        time_used = end_time-start_time
        self.wall_time += time_used
        torch.save(self, os.path.join(self.save_models_dir,'last_epoch.pkl'))
        if save_best:
            torch.save(best_netG, os.path.join(self.save_models_dir, 'best_netG.pkl'))

# ===================================================== #
#               Physics-reinforced GAN                  #
# ===================================================== #
class PICNN():
    def __init__(self, PF_data, net=None, **kwargs):
        
        self.PF_data = PF_data
        self.device = kwargs.get('device',torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.net = (UNet3D().to(self.device) if net==None else net)
        self.loss_func = nn.MSELoss()
        
        self.iter = 0
        self.loss_list = []
        self.fi_list = []
        self.sJ_list = []
        self.L_list = []
        
        self.L_div = 1.e0
        self.L_ff = 1.e0
        self.L_bc = 1.e5
        self.L_sJ = 1.e2
        self.L_fi = 1.e5
        self.L_lf = 1.e3
        
        self.print_epoch = 10
        self.save_epoch = 100
        self._use_mf = False
        
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1.e-3, betas = (0.5, 0.999))
        self.scheduler = StepLR(self.optimizer, step_size=100, gamma=0.9)   
        
        self.save_models_dir = './trainer/3D_models/'
        self.save_pics_dir = './trainer/3D_pics/'   
        
    def train(self, epoch=100, lr=1e-3, Delta=100, gamma=0.9, **kwargs):
        save_best            = kwargs.get('save_best',True)
        save_first           = kwargs.get('save_first', True)
        mf_dt                = kwargs.get('mf_dt', 1.e-5)
        self.save_models_dir = kwargs.get('save_models_dir', self.save_models_dir)
        save_model_name      = kwargs.get('save_model_name', 'PICNN')
        
        if self.iter == 0:
            self.wall_time = 0.
        start_time = time.time()
        
        if lr == 'continue':
            first = True
        else:
            first = True
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        self.scheduler.gamma = gamma
        self.scheduler.step_size = Delta
        
        PF_tensor = torch.from_numpy(self.PF_data).unsqueeze(0).to(self.device).float()
        boundary = torch.from_numpy(self.PF_data[:,:,:,0:1]).to(self.device).float()
        
        for iepoch in range(epoch):
            # update network
            bcube = self.net(PF_tensor)[0]
            fi, sJ, L0 = evaluate(bcube, is_eval=True)
            ff, div, self.sigma_J, self.fi, loss_lf = EQLoss(bcube)
            loss_lf = self.L_lf*loss_lf
            self.loss_div = self.L_div*div
            self.loss_ff = self.L_ff*ff
            self.loss_bc = self.L_bc*self.loss_func(boundary,bcube[:,:,:,0:1])
            self.optimizer.zero_grad()
            self.loss = self.loss_div+self.loss_ff+self.loss_bc+self.L_fi*self.fi+self.L_sJ*self.sigma_J+loss_lf
            self.loss.backward()
            self.loss_list.append(self.loss.item())
            self.fi_list.append(fi)
            self.sJ_list.append(sJ)
            self.L_list.append(L0)
            self.optimizer.step()
            self.scheduler.step()
            
            if self._use_mf:
                bcube = self.net(PF_tensor)[0]
                # print('bcube size:', bcube.size())
                # print('PF size   :', PF_tensor.size())
                bcube[:,:,:,0] = PF_tensor[0,:,:,:,0].to(self.device)
                mf_result = use_opt(bcube, boundary[:,:,:,0], dt=mf_dt)(is_print=False).to(self.device)
                L1 = evaluate(mf_result)[2]
                # if L1<L0:
                self.optimizer.zero_grad()
                loss = self.loss_func(bcube,mf_result)
                loss.backward()
                self.optimizer.step()
                # self.scheduler.step()

            # print training information
            if self.iter % self.print_epoch ==0 or first:
                time_now = time.time()
                time_used = time_now-start_time+self.wall_time
                print(
                'Iter %05d, Loss: %.3e, Loss_div: %.3e,Loss_ff: %.3e, Loss_bc: %.3e, Loss_lf: %.3e, wall_time: %.3e sec, lr: %.3e' % \
                    (self.iter, self.loss_list[-1], self.loss_div, self.loss_ff, self.loss_bc, loss_lf, time_used, 
                     self.scheduler.get_last_lr()[-1])
                )
            
            if self.iter % self.save_epoch == 0 or first:
                if not os.path.exists( self.save_models_dir):
                    os.makedirs( self.save_models_dir)
                    print('create the DIR. : %s' % self.save_models_dir)
                if (self.iter == 0) and save_first:
                    torch.save(self, os.path.join(self.save_models_dir,save_model_name+'_%05d.pkl') % self.iter)
                elif (self.iter !=0):
                    torch.save(self, os.path.join(self.save_models_dir,save_model_name+'_%05d.pkl') % self.iter)
                else:
                    None
            if save_best and self.loss_list != []:
                if self.loss_list[-1]<=np.min(self.loss_list):
                    torch.save(self,os.path.join(self.save_models_dir,'best_model.pkl'))
            
            self.iter+=1
            first = False
            bcube.detach()
            del bcube
            # torch.cuda.empty_cache()
        self.wall_time = time_used
        torch.save(self, os.path.join(self.save_models_dir,save_model_name+'_%05d.pkl') % self.iter)