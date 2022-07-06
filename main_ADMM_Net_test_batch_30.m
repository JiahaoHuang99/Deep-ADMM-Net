%% This is a cpu test code demo for ADMM_Net_v1 reconstruction.
%% Output: the average NMSE and PSNR over the test images.

 clc;
 clear all;
 addpath('./layersfunction/')
 addpath('./util')
 
%% Load trained network
load('./net/network_20/net-stage15.mat')

%% Load mask
load('./mask/mask_20.mat')

%% Load data 
load('./data/imgs30.mat')

%% Save dir
savedir = './data/result30/';

%% 
MSE = [];
PSNR = [];
SSIM = [];

%% Loop
for i=1:30
fprintf('%d/30\n',i)

img=squeeze(img_ori(i,:,:));

%% Undersampling in the k-space
kspace_full = fft2(img); 
y = (double(kspace_full)) .* (ifftshift(mask));
data.train = y;
data.label = img;

%% ZF
zero_filling_rec = ifft2(y);

%% reconstrction by ADMM-Net
[re_LOss, rec_image] = loss_with_gradient_single_before(data, net);

%% evaluation
% re_MSE = mse(abs(rec_image) , abs(data.label))
% re_PSNR = psnr(abs(rec_image) , abs(data.label))
% re_SSIM = ssim(abs(rec_image) , abs(data.label))
% MSE = [MSE, re_MSE];
% PSNR = [PSNR, re_PSNR];
% SSIM = [SSIM, re_SSIM];

%% Save Image
imwrite(abs(img),[savedir, 'GroundTruth/groungtruth_',int2str(i),'.png'])
imwrite(abs(rec_image),[savedir, 'Generated/ADMM_',int2str(i),'.png'])
imwrite(abs(zero_filling_rec),[savedir,'Bad/bad_',int2str(i),'.png'])

end
imwrite(abs(mask),'mask.png')



