%% This is a cpu test code demo for ADMM_Net_v1 reconstruction.
%% Output: the average NMSE and PSNR over the test images.

 clc;
 clear all;
 addpath('./layersfunction/')
 addpath('./util')
 tic
%% Load trained network
% load('./net/network_20/net-stage15.mat')
load('./Train_output/net/net-097.mat')

%% Load mask
% load('./mask/mask_20.mat')
% load('./mask/GaussianDistribution2DMask_30.mat')
% mask = double(maskRS2);
load('./mask/GaussianDistribution1DMask_30.mat')
mask = double(maskRS1);

%% Save dir
savedir = './data/result_G1D30_CC/single';

%% 
MSE = [];
PSNR = [];
SSIM = [];

%% Loop
img_ori = double(imread('./data/result_G1D30_CC/single/sample_comparison.png'))/255;
% img_ori = double(rgb2gray(imread('./data/sample_compare.png')))/255;

img=img_ori;

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
imwrite(abs(img),[savedir, '/ADMM_GT_1095.png'])
imwrite(abs(rec_image),[savedir, '/ADMM_Recon_1095.png'])
imwrite(abs(zero_filling_rec),[savedir,'/ADMM_ZF_1095.png'])


imwrite(abs(mask),'mask.png')

toc

