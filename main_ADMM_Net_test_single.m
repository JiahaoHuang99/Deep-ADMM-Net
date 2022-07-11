%% This is a cpu test code demo for ADMM_Net_v1 reconstruction.
%% Output: the average NMSE and PSNR over the test images.

clc;
clear all;
addpath('./layersfunction/')
addpath('./util')
tic


%% Load trained network

% % load CC G1D10
load('./Train_output_G1D10_CC/net/net-100.mat')

% load CC G1D30
% load('./Train_output_G1D30_CC/net/net-100.mat')

% % load CC G2D30
% load('./Train_output_G2D30_CC/net/net-100.mat')

%% Load mask
% % load G1D10 mask
load('./mask/GaussianDistribution1DMask_10.mat')
mask = double(maskRS1);

% load G1D30 mask
% load('./mask/GaussianDistribution1DMask_30.mat')
% mask = double(maskRS1);

% % load G2D30 mask
% load('./mask/GaussianDistribution2DMask_30.mat')
% mask = double(maskRS2);


%% Save dir
savedir = './data/result_G1D10_CC/ADMM_G1D10_CC_single';
% savedir = './data/result_G1D30_CC/ADMM_G1D30_CC_single';
% savedir = './data/result_G2D30_CC/ADMM_G2D30_CC_single';

if ~exist(savedir,'dir')
    mkdir(savedir); end

%% Read
img_ori = double(imread('./data/sample/GT_01440.png'))/255;
% img_ori = double(rgb2gray(imread('./data/sample/GT_01440.png')))/255;

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


%% Save Image
gt = abs(img);
recon = abs(rec_image);
zf = abs(zero_filling_rec);
imwrite(gt,[savedir, '/ADMM_GT_01440.png'])
imwrite(recon,[savedir, '/ADMM_Recon_01440.png'])
imwrite(zf,[savedir,'/ADMM_ZF_01440.png'])
save([savedir, '/ADMM_GT_01440.mat'], 'gt')
save([savedir, '/ADMM_Recon_01440.mat'], 'recon')
save([savedir, '/ADMM_ZF_01440.mat'], 'zf')

toc

