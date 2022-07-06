function Gen_traindata( )
config;
ND = nnconfig.DataNmber;
%% Load samping pattern 

load('./mask/GaussianDistribution1DMask_30.mat')
mask = double(maskRS1);
% load('./mask/radial_10.mat')
% mask = double(mask);
save(strcat('./mask' , '.mat'), 'mask');

% for i = 1:1:ND 
% dir = './data/ChestTrain/im-';
% load (strcat(dir , saveName(i,floor(log10(ND)))));
% kspace_full = fft2(im_ori); 
% y = (double(kspace_full)) .* (ifftshift(mask));
% data.train = y;
% data.label = im_ori;
% save(strcat('./data/ChestTrain_sampling/', saveName(i, 2), '.mat'), 'data');
% end

%% for CC braindata
for i = 1:1:47
    for j = 1:1:100
    
    load (strcat('./data/Brain_data/db_train_mat/',['imgGT_',num2str(i),'_',num2str(j),'.mat']));
    kspace_full = fft2(double(im_ori)); 
    y = (double(kspace_full)) .* (ifftshift(mask));
    data.train = y;
    data.label = im_ori;

    save(strcat('./data/Brain_data_sampling/', sprintf(saveName((i-1)*100+j, 4))), 'data');
    end
end


%% for fastMRI braindata
% data_path = './data/Knee_data/training/';
% data_names = {dir(fullfile(data_path,'*.mat')).name};
% 
% for idx = 1:1:length(data_names(:))
%     idx
%     slice_path = strcat(data_path,cell2mat(data_names(idx)));
%     load(slice_path);
%     im_ori = imresize(im_ori,[256 256]);
%     kspace_full = fft2(double(im_ori)); 
%     y = (double(kspace_full)) .* (ifftshift(mask));
%     data.train = y;
%     data.label = im_ori;
%     save(strcat('./data/Knee_data_sampling_G1D10/training/', sprintf('data_%05d.mat', idx)), 'data');
% end






