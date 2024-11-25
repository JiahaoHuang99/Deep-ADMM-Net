%% This is a cpu test code demo for ADMM_Net_v1 reconstruction.
%% Output: the average NMSE and PSNR over the test images.

 clc;
 clear all;
 addpath('./layersfunction/')
 addpath('./util')
 
%% Load trained network
<<<<<<< Updated upstream
% load CC G1D10
load('./Train_output/net/net-001.mat')
% load('./Train_output_G1D10_CC/net/net-001.mat')
% % load CC G1D30
% load('./Train_output_G1D30_CC/net/net-097.mat')
% % load CC G1D30 (Old)
% load('./Train_output_G1D30_CC_OLD/net/net-097.mat')
% % load CC G1D30
% load('./Train_output_G2D30_CC_OLD/net/net-100.mat')
=======

% % load CC G1D10
% load('./Train_output_G1D10_CC/net/net-100.mat')

% load CC G1D30
% load('./Train_output_G1D30_CC/net/net-100.mat')

% % load CC G2D30
load('./Train_output_G2D30_CC/net/net-100.mat')
>>>>>>> Stashed changes

%% Load mask
% % load G1D10 mask
% load('./mask/GaussianDistribution1DMask_10.mat')
% mask = double(maskRS1);

% load G1D30 mask
% load('./mask/GaussianDistribution1DMask_30.mat')
% mask = double(maskRS1);

% % load G2D30 mask
load('./mask/GaussianDistribution2DMask_30.mat')
mask = double(maskRS2);

%% Load data 
files = dir('./data/Brain_data/db_valid_mat/*.mat');

%% Save dir
% savedir = './data/result_G1D10_CC/';
% savedir = './data/result_G1D30_CC/';
savedir = './data/result_G2D30_CC/';

%% Init
% MSE = [];
% PSNR = [];
% SSIM = [];

%% Loop
for i=1:2000
    fprintf('%d/2000 %s \n',i,files(i).name)    

    load(fullfile(files(i).folder,files(i).name));

    img = (im_ori - min(min(im_ori)))/(max(max(im_ori)) - min(min(im_ori)));
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
    if ~exist([savedir, 'png/GT/'],'dir')
        mkdir([savedir, 'png/GT/']); end
    if ~exist([savedir, 'png/Recon/'],'dir')
        mkdir([savedir, 'png/Recon/']); end
    if ~exist([savedir, 'png/ZF/'],'dir')
        mkdir([savedir, 'png/ZF/']); end
    if ~exist([savedir, 'mat/GT/'],'dir')
        mkdir([savedir, 'mat/GT/']); end
    if ~exist([savedir, 'mat/Recon/'],'dir')
        mkdir([savedir, 'mat/Recon/']); end
    if ~exist([savedir, 'mat/ZF/'],'dir')
        mkdir([savedir, 'mat/ZF/']); end

    gt = abs(img);
    recon = abs(rec_image);
    zf = abs(zero_filling_rec);
    imwrite(gt,[savedir, 'png/GT/ADMM_GT_',int2str(i),'.png'])
    imwrite(recon,[savedir, 'png/Recon/ADMM_Recon_',int2str(i),'.png'])
    imwrite(zf,[savedir,'png/ZF/ADMM_ZF_',int2str(i),'.png'])
    save([savedir, 'mat/GT/ADMM_GT_',int2str(i),'.mat'], 'gt')
    save([savedir, 'mat/Recon/ADMM_Recon_',int2str(i),'.mat'], 'recon')
    save([savedir, 'mat/ZF/ADMM_ZF_',int2str(i),'.mat'], 'zf')

end
imwrite(abs(mask),[savedir, 'mask.png'])



