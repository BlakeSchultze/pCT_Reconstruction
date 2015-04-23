%% Hull Comparison
close all;

Phantom = importdata('NeoZdim200.txt');
Phantom = mat2gray(Phantom);
figure, imshow(Phantom);
Phantom_Dimensions = size(Phantom);

Data_Set = '(sim)';
% Data_Set = '(noisy)';
% Data_Set = '(rat)';
% Data_Set = '(ped)';

File_FBP = 'x_FBP';
File_SC = 'x_sc';
File_MSC = 'x_msc';
File_SM = 'x_sm';
File_BC = 'bin_carve';

% % Filtered Back-Projection (FBP) Hull Comparison
% FILE_FBP = [File_FBP, Data_Set,'.txt'];
% slice_FBP = 5; %[1,end]
% MANUAL_RESIZE_THRESH_FBP = true;
% avg_thresh_FBP = 0.1;
% if(MANUAL_RESIZE_THRESH_FBP == true)
% 	resize_thresh_FBP = 0;
% 	[missing_FBP, extra_FBP, Hull_Compared_FBP] = Phantom_Hull_Comparison( Phantom, FILE_FBP, slice_FBP, true, 'both', resize_thresh_FBP,avg_thresh_FBP );
% else
%     [missing_FBP, extra_FBP, Hull_Compared_FBP] = Phantom_Hull_Comparison( Phantom, FILE_FBP, slice_FBP, true, avg_thresh_FBP);
% end
% %Hull_min_FBP, Hull_max_FBP, resize_thresh_FBP,
% missing_FBP, extra_FBP
% figure, imshow(Hull_Compared_FBP)
% title = [ filename, num2str(slice_FBP), '(',num2str(avg_thresh_FBP),').png' ];
% imwrite(Hull_Compared_FBP , title, 'png', 'bitdepth', 2 )

% % Space Carving (SC) Hull Comparison
% FILE_SC = [File_SC, Data_Set,'.txt'];
% slice_SC = 5; %[1,end]
% MANUAL_RESIZE_THRESH_SC = true;
% avg_thresh_SC = 0.0;
% if(MANUAL_RESIZE_THRESH_SC == true)
% 	resize_thresh_SC = 0;
% 	[missing_SC, extra_SC, Hull_Compared_SC] = Phantom_Hull_Comparison( Phantom, FILE_SC , slice_SC, true, 'both', resize_thresh_SC, avg_thresh_SC );
% else
%     [missing_SC, extra_SC, Hull_Compared_SC] = Phantom_Hull_Comparison( Phantom, FILE_SC , slice_SC, true, avg_thresh_SC);
% end
% %Hull_min_SC, Hull_max_SC, resize_thresh_SC,
% missing_SC, extra_SC
% figure, imshow(Hull_Compared_SC)
% title = [ filename, num2str(slice_SC), '(',num2str(avg_thresh_SC),').png' ];
%imwrite(Hull_Compared_SC , title, 'png', 'bitdepth', 2 )

% Modified Space Carving (MSC) Hull Comparison
FILE_MSC = [File_MSC, Data_Set,'.txt'];
slice_MSC = 5; %[1,end]
MANUAL_RESIZE_THRESH_MSC = true;
avg_thresh_MSC = 0.1;
if(MANUAL_RESIZE_THRESH_MSC == true)
	resize_thresh_MSC = 0;
	[missing_MSC, extra_MSC, Hull_Compared_MSC] = Phantom_Hull_Comparison( Phantom, FILE_MSC, slice_MSC, true, 'both', resize_thresh_MSC, avg_thresh_MSC );
else
    [missing_MSC, extra_MSC, Hull_Compared_MSC] = Phantom_Hull_Comparison( Phantom, FILE_MSC, slice_MSC, true, avg_thresh_MSC);
end
%Hull_min_MSC, Hull_max_MSC, resize_thresh_MSC,
missing_MSC, extra_MSC
figure, imshow(Hull_Compared_MSC)
title = [ filename, num2str(slice_MSC), '(',num2str(avg_thresh_MSC),').png' ];
%imwrite(Hull_Compared_MSC , title, 'png', 'bitdepth', 2 )

% Space Modeling (SM) Hull Comparison
FILE_SM = [File_SM, Data_Set,'.txt'];
slice_SM = 20; %[1,end]
MANUAL_RESIZE_THRESH_SM = true;
avg_thresh_SM = 0.4;
if(MANUAL_RESIZE_THRESH_SM == true)
	resize_thresh_SM = 0;
	[missing_SM, extra_SM, Hull_Compared_SM] = Phantom_Hull_Comparison( Phantom, FILE_SM, slice_SM, true, 'both', resize_thresh_SM, avg_thresh_SM );
else
    [missing_SM, extra_SM, Hull_Compared_SM] = Phantom_Hull_Comparison( Phantom, FILE_SM, slice_SM, true, avg_thresh_SM);
end
%Hull_min_SM, Hull_max_SM, resize_thresh_SM,
missing_SM, extra_SM
figure, imshow(Hull_Compared_SM)
title = [ filename, num2str(slice_SM), '(',num2str(avg_thresh_SM),').png' ];
%imwrite(Hull_Compared_SM , title, 'png', 'bitdepth', 2 )

% Bin Carving (BC) Hull Comparison
FILE_BC = [File_BC, Data_Set,'.txt'];
slice_BC = 32; %[1,end]
MANUAL_RESIZE_THRESH_BC = true;
avg_thresh_BC = 0.1;
if(MANUAL_RESIZE_THRESH_BC == true)
	resize_thresh_BC = 0;
	[missing_BC, extra_BC, Hull_Compared_BC, Filtered_Hull_BC] = Phantom_Hull_Comparison( Phantom, FILE_BC, slice_BC, true, 'both', resize_thresh_BC, avg_thresh_BC );
else
    [missing_BC, extra_BC, Hull_Compared_BC, Filtered_Hull_BC] = Phantom_Hull_Comparison( Phantom, FILE_BC, slice_BC, true, avg_thresh_BC);
end
%Hull_min_BC, Hull_max_BC, resize_thresh_BC,
missing_BC, extra_BC
figure, imshow(Filtered_Hull_BC)
figure, imshow(Hull_Compared_BC)
title = [ filename, num2str(slice_BC), '(',num2str(avg_thresh_BC),').png' ];
%imwrite(Hull_Compared_BC , title, 'png', 'bitdepth', 2 )

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

title = [ filename, num2str(slice_SM), '(',num2str(avg_thresh_SM),').png' ];
%imwrite(Processed_Hull_SM , title, 'png', 'bitdepth', 2 )


