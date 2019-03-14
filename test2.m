clear all; close all; clc;
 
% Read an image
%A = imread('lena.jpg');
 
% Translate the image
%A_trans = imtranslate(A, [5 5]);
 
% Write the transformed image to disk
%imwrite(A_trans, 'newlena.jpg');
 
%figure,imshow(A_trans);  ¸ÄÍ¼Æ¬ÏñËØÎª28*28

I=imread('4.png');
J=imresize(I,[28,28]);
imshow(I);
figure;
imshow(J);
imwrite(J,'new4.png');
