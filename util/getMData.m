function data = getMData (n)
config;
size = nnconfig.ImageSize;
ND = nnconfig.DataNmber;
data.train = single(zeros(size));
data.label = single (zeros(size));
dir = nnconfig.DataPath;
% load (strcat(dir , saveName(n,ceil(log10(ND)))));
load (strcat(dir , saveName(n,4)));
