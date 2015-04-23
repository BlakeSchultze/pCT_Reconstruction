%% Hull Comparison
close all;

 Data_Set = '(rat)';
% Data_Set = '(ped)';

File_FBP = 'x_FBP';
File_SC = 'x_sc';
File_MSC = 'x_msc';
File_SM = 'x_sm';
File_BC = 'bin_carve';


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


% Space Modeling (SM) Hull Comparison
Noisy:19, Sim: 20, Rat: , Ped: 
FILE_SM = [File_SM, Data_Set,'.txt'];
slice_SM = 7; %[1,end]
Hull_SM = importdata(FILE_SM);
[rows, cols] = size(Hull_SM)
avg_thresh_SM = 0.1;
Hull_Slice_SM = Return_Hull_Slice(Hull_SM, slice_SM);
%[Processed_Hull_SM, Filtered_Hull_SM] = Post_Process_Hull(Hull_Slice_SM, avg_thresh_SM);
%figure, imshow(Filtered_Hull_SM);
%figure, imshow(Processed_Hull_SM);

title = [ filename, num2str(slice_SM), '(',num2str(avg_thresh_SM),').png' ];
%imwrite(Processed_Hull_SM , title, 'png', 'bitdepth', 2 )