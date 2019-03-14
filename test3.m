clear all;close all;clc;
% Read an input image
A = imread('new4.png');
 
% Convert the image to single-channel grayscale image
A_gray = rgb2gray(A);
 
figure,imhist(A_gray),title('hist of A_grey');
 
% Convert image to double i.e., [0,1]
A_gray = im2double(A_gray);
 
% Generate threhold value using Otsu's algorithm
otsu_level = graythresh(A_gray);
 
% Threshold image using Otsu's threshold and manually defined
% threshold values
B_otsu_thresh = im2bw(A_gray, otsu_level);
B_thresh_50 = im2bw(A_gray, 50/255);
B_thresh_100 = im2bw(A_gray, 100/255);
B_thresh_150 = im2bw(A_gray, 150/255);
B_thresh_200 = im2bw(A_gray, 200/255);
 
% Display original and thresholded binary images side-by-side
figure, subplot(2, 3, 1), imshow(A_gray), title('Original image');
subplot(2, 3, 2), imshow(B_otsu_thresh), title('Binary image using Otsu threshold value');
subplot(2, 3, 3), imshow(B_thresh_50), title('Binary image using threshold value = 50');
subplot(2, 3, 4), imshow(B_thresh_100), title('Binary image using threshold value = 100');
subplot(2, 3, 5), imshow(B_thresh_150), title('Binary image using threshold value = 150');
subplot(2, 3, 6), imshow(B_thresh_200), title('Binary image using threshold value = 200');
imwrite(B_otsu_thresh,'newnew4.png');