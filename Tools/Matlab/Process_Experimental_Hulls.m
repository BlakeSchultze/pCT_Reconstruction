%% Hull Comparison
close all;

 Data_Set = '(rat)';
% Data_Set = '(ped)';

File_FBP = 'x_FBP';
File_SC = 'x_sc';
File_MSC = 'x_msc';
File_SM = 'x_sm';
File_BC = 'bin_carve';

% % Filtered Back-Projection (FBP) Hull Comparison
% FILE_FBP = [File_FBP, Data_Set,'.txt'];
% slice_FBP = 7; %[1,end]
% Hull_FBP = importdata(FILE_FBP);
% [rows, cols] = size(Hull_FBP)
% avg_thresh_FBP = 0.1;
% Hull_Slice_FBP = Return_Hull_Slice(Hull_FBP, slice_FBP);
% [Processed_Hull_FBP, Filtered_Hull_FBP] = Post_Process_Hull(Hull_Slice_FBP, avg_thresh_FBP);
% figure, imshow(Filtered_Hull_FBP);
% figure, imshow(Processed_Hull_FBP);
% title = [ filename, num2str(slice_FBP), '(',num2str(avg_thresh_FBP),').png' ];
% imwrite(Processed_Hull_FBP , title, 'png', 'bitdepth', 2 )
 
% % Space Carving (SC) Hull Comparison
% FILE_SC = [File_SC, Data_Set,'.txt'];
% slice_SC = 7; %[1,end]
% Hull_SC = importdata(FILE_SC);
% [rows, cols] = size(Hull_SC)
% avg_thresh_SC = 0.1;
% Hull_Slice_SC = Return_Hull_Slice(Hull_SC, slice_SC);
% [Processed_Hull_SC, Filtered_Hull_SC] = Post_Process_Hull(Hull_Slice_SC, avg_thresh_SC);
% figure, imshow(Filtered_Hull_SC);
% figure, imshow(Processed_Hull_SC);
% title = [ filename, num2str(slice_SC), '(',num2str(avg_thresh_SC),').png' ];
% imwrite(Processed_Hull_SC , title, 'png', 'bitdepth', 2 )

% % Modified Space Carving (MSC) Hull Comparison
% FILE_MSC = [File_MSC, Data_Set,'.txt'];
% slice_MSC = 7; %[1,end]
% Hull_MSC = importdata(FILE_MSC);
% [rows, cols] = size(Hull_MSC)
% avg_thresh_MSC = 0.1;
% Hull_Slice_MSC = Return_Hull_Slice(Hull_MSC, slice_MSC);
% [Processed_Hull_MSC, Filtered_Hull_MSC] = Post_Process_Hull(Hull_Slice_MSC, avg_thresh_MSC);
% figure, imshow(Filtered_Hull_MSC);
% figure, imshow(Processed_Hull_MSC);
% title = [ filename, num2str(slice_MSC), '(',num2str(avg_thresh_MSC),').png' ];
% imwrite(Processed_Hull_MSC , title, 'png', 'bitdepth', 2 )


% % Space Modeling (SM) Hull Comparison
% Noisy:19, Sim: 20, Rat: , Ped: 
% FILE_SM = [File_SM, Data_Set,'.txt'];
% slice_SM = 7; %[1,end]
% Hull_SM = importdata(FILE_SM);
% [rows, cols] = size(Hull_SM)
% avg_thresh_SM = 0.1;
% Hull_Slice_SM = Return_Hull_Slice(Hull_SM, slice_SM);
% [Processed_Hull_SM, Filtered_Hull_SM] = Post_Process_Hull(Hull_Slice_SM, avg_thresh_SM);
% figure, imshow(Filtered_Hull_SM);
% figure, imshow(Processed_Hull_SM);
% title = [ filename, num2str(slice_SM), '(',num2str(avg_thresh_SM),').png' ];
% imwrite(Processed_Hull_SM , title, 'png', 'bitdepth', 2 )

% Bin Carving (BC) Hull Comparison
FILE_BC = [File_BC, Data_Set,'.txt'];
slice_BC = 7; %[1,end]
Hull_BC = importdata(FILE_BC);
[rows, cols] = size(Hull_BC)
avg_thresh_BC = 0.4;
Hull_Slice_BC = Return_Hull_Slice(Hull_BC, slice_BC);
[Processed_Hull_BC, Filtered_Hull_BC] = Post_Process_Hull(Hull_Slice_BC, avg_thresh_BC);
figure, imshow(Hull_Slice_BC);
figure, imshow(Filtered_Hull_BC);
figure, imshow(Processed_Hull_BC);
title = [ filename, num2str(slice_BC), '(',num2str(avg_thresh_BC),').png' ];
imwrite(Processed_Hull_BC , title, 'png', 'bitdepth', 2 )




