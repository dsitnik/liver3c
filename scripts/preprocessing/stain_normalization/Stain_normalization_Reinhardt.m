% Stain_normalization_Reinhardt

clear
close all

addpath(genpath('.\Reinhard_StainNormalization'));

target = imread('TARGET_image','tif');

source = imread('CCA','jpg');
norm_img=stainnorm_reinhard(source,target);
imgName = strcat('CCA_sn.jpg');
imwrite(norm_img,imgName,'jpg');


