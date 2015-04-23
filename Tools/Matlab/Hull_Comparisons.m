%% Hull Comparisons
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

% Filtered Back-Projection (FBP) Hull Comparison
slice_FBP = 5; %[1,end]
MANUAL_RESIZE_THRESHOLD_FBP = true;
if(MANUAL_RESIZE_THRESHOLD_FBP == true)
	resize_threshold_FBP = 0;
	[missing_FBP, extra_FBP, Hull_Resized_FBP, Hull_min_FBP, Hull_max_FBP] = Phantom_Hull_Comparison( Phantom, [File_FBP, Data_Set,'.txt'], slice_FBP, resize_threshold_FBP );
else
    [missing_FBP, extra_FBP, Hull_Resized_FBP, Hull_min_FBP, Hull_max_FBP, resize_threshold_FBP ] = Phantom_Hull_Comparison( Phantom, [File_FBP, Data_Set,'.txt'], slice_FBP);
end
%Hull_min_FBP, Hull_max_FBP, resize_threshold_FBP,
missing_FBP, extra_FBP
figure, imshow(Hull_Resized_FBP)

% Space Carving (SC) Hull Comparison
slice_SC = 5; %[1,end]
MANUAL_RESIZE_THRESHOLD_SC = true;
if(MANUAL_RESIZE_THRESHOLD_SC == true)
	resize_threshold_SC = 0;
	[missing_SC, extra_SC, Hull_Resized_SC, Hull_min_SC, Hull_max_SC] = Phantom_Hull_Comparison( Phantom, [File_SC, Data_Set,'.txt'], slice_SC, resize_threshold_SC );
else
    [missing_SC, extra_SC, Hull_Resized_SC, Hull_min_SC, Hull_max_SC, resize_threshold_SC ] = Phantom_Hull_Comparison( Phantom, [File_SC, Data_Set,'.txt'], slice_SC);
end
%Hull_min_SC, Hull_max_SC, resize_threshold_SC,
missing_SC, extra_SC
figure, imshow(Hull_Resized_SC)

% Modified Space Carving (MSC) Hull Comparison
slice_MSC = 5; %[1,end]
MANUAL_RESIZE_THRESHOLD_MSC = true;
if(MANUAL_RESIZE_THRESHOLD_MSC == true)
	resize_threshold_MSC = 0;
	[missing_MSC, extra_MSC, Hull_Resized_MSC, Hull_min_MSC, Hull_max_MSC] = Phantom_Hull_Comparison( Phantom, [File_MSC, Data_Set,'.txt'], slice_MSC, resize_threshold_MSC );
else
    [missing_MSC, extra_MSC, Hull_Resized_MSC, Hull_min_MSC, Hull_max_MSC, resize_threshold_MSC ] = Phantom_Hull_Comparison( Phantom, [File_MSC, Data_Set,'.txt'], slice_MSC);
end
%Hull_min_MSC, Hull_max_MSC, resize_threshold_MSC,
missing_MSC, extra_MSC
figure, imshow(Hull_Resized_MSC)

% Space Modeling (SM) Hull Comparison
slice_SM = 5; %[1,end]
MANUAL_RESIZE_THRESHOLD_SM = true;
if(MANUAL_RESIZE_THRESHOLD_SM == true)
	resize_threshold_SM = 0;
	[missing_SM, extra_SM, Hull_Resized_SM, Hull_min_SM, Hull_max_SM] = Phantom_Hull_Comparison( Phantom, [File_SM, Data_Set,'.txt'], slice_SM, resize_threshold_SM );
else
    [missing_SM, extra_SM, Hull_Resized_SM, Hull_min_SM, Hull_max_SM, resize_threshold_SM ] = Phantom_Hull_Comparison( Phantom, [File_SM, Data_Set,'.txt'], slice_SM);
end
%Hull_min_SM, Hull_max_SM, resize_threshold_SM,
missing_SM, extra_SM
figure, imshow(Hull_Resized_SM)

%% Post Processed Hull Comparisons
close all;

Phantom = importdata('NeoZdim200.txt');
Phantom = mat2gray(Phantom);
figure, imshow(Phantom);
Phantom_Dimensions = size(Phantom);

% Data_Set = '(sim)';
% Data_Set = '(noisy)';
 Data_Set = '(rat)';
% Data_Set = '(ped)';

File_FBP = 'x_FBP';
File_SC = 'x_sc';
File_MSC = 'x_msc';
File_PC_MSC = 'post_cut_x_msc';
File_SM = 'x_sm';
File_PC_SM = 'post_cut_x_sm';
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

% % Modified Space Carving (MSC) Hull Comparison
% FILE_MSC = [File_MSC, Data_Set,'.txt'];
% slice_MSC = 33; %[1,end]
% MANUAL_RESIZE_THRESH_MSC = true;
% avg_thresh_MSC = 0.2;
% if(MANUAL_RESIZE_THRESH_MSC == true)
% 	resize_thresh_MSC = 0;
% 	[missing_MSC, extra_MSC, Hull_Compared_MSC] = Phantom_Hull_Comparison( Phantom, FILE_MSC, slice_MSC, true, 'both', resize_thresh_MSC, avg_thresh_MSC );
% else
%     [missing_MSC, extra_MSC, Hull_Compared_MSC] = Phantom_Hull_Comparison( Phantom, FILE_MSC, slice_MSC, true, avg_thresh_MSC);
% end
% %Hull_min_MSC, Hull_max_MSC, resize_thresh_MSC,
% missing_MSC, extra_MSC
% figure, imshow(Hull_Compared_MSC)
% title = [ File_MSC, Data_Set, num2str(slice_MSC), '(',num2str(avg_thresh_MSC),').png' ];
% imwrite(Hull_Compared_MSC , title, 'png', 'bitdepth', 2 )

% %18, 4563, 0.87
% % Space Modeling (SM) Hull Comparison
% FILE_SM = [File_SM, Data_Set,'.txt'];
% slice_SM = 21; %[1,end]
% MANUAL_RESIZE_THRESH_SM = true;
% avg_thresh_SM = 0.99;
% if(MANUAL_RESIZE_THRESH_SM == true)
% 	resize_thresh_SM = 0;
% 	[missing_SM, extra_SM, Hull_Compared_SM] = Phantom_Hull_Comparison( Phantom, FILE_SM, slice_SM, true, 'both', resize_thresh_SM, avg_thresh_SM );
% else
%     [missing_SM, extra_SM, Hull_Compared_SM] = Phantom_Hull_Comparison( Phantom, FILE_SM, slice_SM, true, avg_thresh_SM);
% end
% %Hull_min_SM, Hull_max_SM, resize_thresh_SM,
% missing_SM, extra_SM
% figure, imshow(Hull_Compared_SM)
% title_SM = [ 'pre_cut_',File_SM, Data_Set, num2str(slice_SM), '(',num2str(avg_thresh_SM),').png' ]
% imwrite(Hull_Compared_SM , title_SM, 'png', 'bitdepth', 2 )
 
% %18, 4563, 0.87
% % Post-Cut Space Modeling (PCSM) Hull Comparison
% FILE_PC_SM = [File_PC_SM, Data_Set,'.txt'];
% slice_PC_SM = 18; %[1,end]
% MANUAL_RESIZE_THRESH_PC_SM = true;
% avg_thresh_PC_SM = 0.87;
% if(MANUAL_RESIZE_THRESH_PC_SM == true)
% 	resize_thresh_PC_SM = 0;
% 	[missing_PC_SM, extra_PC_SM, Hull_Compared_PC_SM] = Phantom_Hull_Comparison( Phantom, FILE_PC_SM, slice_PC_SM, true, 'both', resize_thresh_PC_SM, avg_thresh_PC_SM );
% else
%     [missing_PC_SM, extra_PC_SM, Hull_Compared_PC_SM] = Phantom_Hull_Comparison( Phantom, FILE_PC_SM, slice_PC_SM, true, avg_thresh_PC_SM);
% end
% %Hull_min_PC_SM, Hull_max_PC_SM, resize_thresh_PC_SM,
% missing_PC_SM, extra_PC_SM
% figure, imshow(Hull_Compared_SM)
% title_PC_SM = [ File_PC_SM, Data_Set, num2str(slice_PC_SM), '(',num2str(avg_thresh_PC_SM),').png' ]
% %imwrite(Hull_Compared_PC_SM , title_PC_SM, 'png', 'bitdepth', 2 )

% Bin Carving (BC) Hull Comparison
FILE_BC = [File_BC, Data_Set,'.txt'];
slice_BC = 7; %[1,end]
MANUAL_RESIZE_THRESH_BC = true;
avg_thresh_BC = 0.4;
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
title_BC = [ File_BC, Data_Set num2str(slice_BC), '(',num2str(avg_thresh_BC),').png' ];
% imwrite(Hull_Compared_BC , title_BC, 'png', 'bitdepth', 2 )

%% No Processing Hull Comparisons
close all;

% Data_Set = '(sim)';
% Data_Set = '(noisy)';
 Data_Set = '(rat)';
% Data_Set = '(ped)';

File_FBP = 'x_FBP';
File_SC = 'x_sc';
File_MSC = 'x_msc';
File_SM = 'x_sm';
File_BC = 'bin_carve';

Phantom = importdata('NeoZdim200.txt');
Phantom = mat2gray(Phantom);
figure, imshow(Phantom);
Phantom_Dimensions = size(Phantom);

% % Filtered Back Projection (FBP) Hull Comparison
% % Noisy:19, Sim: 20, Rat: , Ped: 
% FILE_FBP = [File_FBP, Data_Set, '.txt']
% slice_FBP = 5; %[1,end]
% [missing_FBP, extra_FBP, Hull_Compared_FBP] = Phantom_Hull_Comparison( Phantom, FILE_FBP, slice_FBP, false );
% missing_FBP, extra_FBP
% title = [ File_FBP, Data_Set, num2str(slice_FBP),'.png' ];
% figure, imshow(Hull_Compared_FBP)
% imwrite(Hull_Compared_FBP , title, 'png', 'bitdepth', 2 )

% % Modified Space Carving (MSC) Hull Comparison
% % Noisy:19, Sim: 20, Rat: , Ped: 
% FILE_MSC = [File_MSC, Data_Set, '.txt'];
% slice_MSC = 5; %[1,end]
% [missing_MSC, extra_MSC, Hull_Compared_MSC] = Phantom_Hull_Comparison( Phantom, FILE_MSC, slice_MSC, false);
% missing_MSC, extra_MSC
% title = [ File_MSC, Data_Set, num2str(slice_MSC),'.png' ];
% %imwrite(Processed_Hull_MSC , title, 'png', 'bitdepth', 2 )


% % Space Modeling (SM) Hull Comparison
% % Noisy:19, Sim: 20, Rat: , Ped: 
% FILE_SM = [File_SM, Data_Set,'.txt'];
% slice_SM = 20; %[1,end]
% [missing_SM, extra_SM, Hull_Compared_SM] = Phantom_Hull_Comparison( Phantom, FILE_SM, slice_SM, false );
% missing_SM, extra_SM
% title = [ File_SM, Data_Set, num2str(slice_SM), '(',num2str(avg_thresh_SM),').png' ];
% %imwrite(Processed_Hull_SM , title, 'png', 'bitdepth', 2 )

%% Pre and Post Cut Hull Comparison

close all;

 Data_Set = '(sim)';
% Data_Set = '(noisy)';

File_MSC = 'x_msc';
File_PC_MSC = 'post_cut_x_msc';
File_SM = 'x_sm';
File_PC_SM = 'post_cut_x_sm';

% % Modified Space Carving (MSC) Hull Comparison
% % Noisy:19, Sim: 20, Rat: , Ped: 
% FILE_MSC = [File_MSC, Data_Set, '.txt'];
% FILE_PC_MSC = [File_PC_MSC, Data_Set, '.txt'];
% slice_MSC = 35; %[1,end]
% [missing_MSC, extra_MSC, Hull_Compared_MSC, Hull_Compared_PC_MSC] = Hull_Hull_Comparison( FILE_MSC, FILE_PC_MSC, slice_MSC);
% missing_MSC, extra_MSC
% figure, imshow(Hull_Compared_MSC);
% figure, imshow(Hull_Compared_PC_MSC);
% title_MSC = [ File_MSC, Data_Set, num2str(slice_MSC),'.png' ];
% title_PC_MSC = [ File_PC_MSC, Data_Set, num2str(slice_MSC),'.png' ];
% imwrite(Hull_Compared_MSC , title_MSC, 'png', 'bitdepth', 2 )
% imwrite(Hull_Compared_PC_MSC , title_PC_MSC, 'png', 'bitdepth', 2 )

% Space Modeling (SM) Hull Comparison
FILE_SM = [File_SM, Data_Set, '.txt'];
FILE_PC_SM = [File_PC_SM, Data_Set, '.txt'];
slice_SM = 37; %[1,end]
[missing_SM, extra_SM, Hull_Compared_SM, Hull_Compared_PC_SM] = Hull_Hull_Comparison( FILE_SM, FILE_PC_SM, slice_SM);
missing_SM, extra_SM, missing_SM + extra_SM
figure, imshow(Hull_Compared_SM);
figure, imshow(Hull_Compared_PC_SM);
title_SM = [ File_SM, Data_Set, num2str(slice_SM),'.png' ];
title_PC_SM = [ File_PC_SM, Data_Set, num2str(slice_SM),'.png' ];
imwrite(Hull_Compared_SM , title_SM, 'png', 'bitdepth', 2 )
imwrite(Hull_Compared_PC_SM , title_PC_SM, 'png', 'bitdepth', 2 )

%% Pre vs Post Cut Post Processed Hull Comparisons
close all;

Phantom = importdata('NeoZdim200.txt');
Phantom = mat2gray(Phantom);
figure, imshow(Phantom);
Phantom_Dimensions = size(Phantom);

 Data_Set = '(sim)';
% Data_Set = '(noisy)';

File_MSC = 'x_msc';
File_PC_MSC = 'post_cut_x_msc';
File_SM = 'x_sm';
File_PC_SM = 'post_cut_x_sm';

% Pre Cut Modified Space Carving (MSC) Hull Comparison
slice_MSC_compared = 5;
FILE_MSC = [File_MSC, Data_Set,'.txt'];
MANUAL_RESIZE_THRESH_MSC = true;
avg_thresh_MSC = 0.2;
if(MANUAL_RESIZE_THRESH_MSC == true)
	resize_thresh_MSC = 0;
	[missing_MSC, extra_MSC, Hull_Compared_MSC] = Phantom_Hull_Comparison( Phantom, FILE_MSC, slice_MSC_compared, true, 'both', resize_thresh_MSC, avg_thresh_MSC );
else
    [missing_MSC, extra_MSC, Hull_Compared_MSC] = Phantom_Hull_Comparison( Phantom, FILE_MSC, slice_MSC_compared, true, avg_thresh_MSC);
end
%Hull_min_MSC, Hull_max_MSC, resize_thresh_MSC,
missing_MSC, extra_MSC
figure, imshow(Hull_Compared_MSC)
title_MSC = [ 'pre_cut_',File_MSC, Data_Set, num2str(slice_MSC_compared), '(',num2str(avg_thresh_MSC),').png' ]
%imwrite(Hull_Compared_MSC , title_MSC, 'png', 'bitdepth', 2 )
 
%18, 4563, 0.87
% Post-Cut Modified Space Carving (PCMSC) Hull Comparison
FILE_PC_MSC = [File_PC_MSC, Data_Set,'.txt'];
MANUAL_RESIZE_THRESH_PC_MSC = true;
avg_thresh_PC_MSC = 0.2;
if(MANUAL_RESIZE_THRESH_PC_MSC == true)
	resize_thresh_PC_MSC = 0;
	[missing_PC_MSC, extra_PC_MSC, Hull_Compared_PC_MSC] = Phantom_Hull_Comparison( Phantom, FILE_PC_MSC, slice_MSC_compared, true, 'both', resize_thresh_PC_MSC, avg_thresh_PC_MSC );
else
    [missing_PC_MSC, extra_PC_MSC, Hull_Compared_PC_MSC] = Phantom_Hull_Comparison( Phantom, FILE_PC_MSC, slice_MSC_compared, true, avg_thresh_PC_MSC);
end
%Hull_min_PC_MSC, Hull_max_PC_MSC, resize_thresh_PC_MSC,
missing_PC_MSC, extra_PC_MSC,
figure, imshow(Hull_Compared_PC_MSC)
title_PC_MSC = [ File_PC_MSC, Data_Set, num2str(slice_MSC_compared), '(',num2str(avg_thresh_PC_MSC),').png' ]
%imwrite(Hull_Compared_PC_MSC , title_PC_MSC, 'png', 'bitdepth', 2 )
missing_MSC - missing_PC_MSC
extra_MSC - extra_PC_MSC

%3,0,11
% slice_SM_compared = 20;
% %18, 4563, 0.87
% % Space Modeling (SM) Hull Comparison
% FILE_SM = [File_SM, Data_Set,'.txt'];
% MANUAL_RESIZE_THRESH_SM = true;
% avg_thresh_SM = 0.4;
% if(MANUAL_RESIZE_THRESH_SM == true)
% 	resize_thresh_SM = 0;
% 	[missing_SM, extra_SM, Hull_Compared_SM] = Phantom_Hull_Comparison( Phantom, FILE_SM, slice_SM_compared, true, 'both', resize_thresh_SM, avg_thresh_SM );
% else
%     [missing_SM, extra_SM, Hull_Compared_SM] = Phantom_Hull_Comparison( Phantom, FILE_SM, slice_SM_compared, true, avg_thresh_SM);
% end
% %Hull_min_SM, Hull_max_SM, resize_thresh_SM,
% missing_SM, extra_SM
% figure, imshow(Hull_Compared_SM)
% title_SM = [ 'pre_cut_',File_SM, Data_Set, num2str(slice_SM_compared), '(',num2str(avg_thresh_SM),').png' ]
% imwrite(Hull_Compared_SM , title_SM, 'png', 'bitdepth', 2 )
%  
% %18, 4563, 0.87
% % Post-Cut Space Modeling (PCSM) Hull Comparison
% FILE_PC_SM = [File_PC_SM, Data_Set,'.txt'];
% MANUAL_RESIZE_THRESH_PC_SM = true;
% avg_thresh_PC_SM = 0.39;
% if(MANUAL_RESIZE_THRESH_PC_SM == true)
% 	resize_thresh_PC_SM = 0;
% 	[missing_PC_SM, extra_PC_SM, Hull_Compared_PC_SM] = Phantom_Hull_Comparison( Phantom, FILE_PC_SM, slice_SM_compared, true, 'both', resize_thresh_PC_SM, avg_thresh_PC_SM );
% else
%     [missing_PC_SM, extra_PC_SM, Hull_Compared_PC_SM] = Phantom_Hull_Comparison( Phantom, FILE_PC_SM, slice_SM_compared, true, avg_thresh_PC_SM);
% end
% %Hull_min_PC_SM, Hull_max_PC_SM, resize_thresh_PC_SM,
% missing_PC_SM, extra_PC_SM,
% figure, imshow(Hull_Compared_SM)
% title_PC_SM = [ File_PC_SM, Data_Set, num2str(slice_SM_compared), '(',num2str(avg_thresh_PC_SM),').png' ]
% imwrite(Hull_Compared_PC_SM , title_PC_SM, 'png', 'bitdepth', 2 )
% missing_SM - missing_PC_SM
% extra_SM - extra_PC_SM
%%
close all;
% Data_Set = '(sim)';
% Data_Set = '(noisy)';
 Data_Set = '(rat)';
% Data_Set = '(ped)';

File_FBP = 'x_FBP';
File_SC = 'x_MSC';
File_MSC = 'x_msc';
File_SM = 'x_sm';
File_BC = 'bin_carve';

% FILE_FBP = [File_FBP, Data_Set, '.txt'];
% x_FBP = importdata(FILE_FBP);
% slice_FBP = 2;
% Hull_Slice_FBP = Return_Hull_Slice(x_FBP, slice_FBP);
% figure, imshow(Hull_Slice_FBP)
% title_FBP = [ File_FBP, Data_Set, num2str(slice_FBP),'.png' ];
% imwrite(Hull_Slice_FBP , title_FBP, 'png', 'bitdepth', 2 )

FILE_BC = [File_BC, Data_Set, '.txt'];
x_BC = importdata(FILE_BC);
slice_BC = 7;
Hull_Slice_BC = Return_Hull_Slice(x_BC, slice_BC);
figure, imshow(Hull_Slice_BC)
%figure, imshow(Hull_Slice_BC)
avg_thresh_BC = 0.4
[Processed_Hull_BC, filtered_Hull_BC] = Post_Process_Hull(Hull_Slice_BC, avg_thresh_BC);
figure, imshow(filtered_Hull_BC)
figure, imshow(Processed_Hull_BC)
title_orig_BC = [ File_BC, Data_Set,'orig', num2str(slice_BC),'.png' ];
title_filt_BC = [ File_BC, Data_Set,'filt', num2str(slice_BC),'.png' ];
title_final_BC = [ File_BC, Data_Set,'final', num2str(slice_BC),'.png' ];
imwrite(Hull_Slice_BC , title_orig_BC, 'png', 'bitdepth', 2 )
imwrite(filtered_Hull_BC , title_filt_BC, 'png', 'bitdepth', 2 )
imwrite(Processed_Hull_BC , title_final_BC, 'png', 'bitdepth', 2 )

% FILE_MSC = [File_MSC, '.txt'];
% %FILE_MSC = [File_MSC, Data_Set, '.txt'];
% x_MSC = importdata(FILE_MSC);
% slice_MSC = 31;
% Hull_Slice_MSC = Return_Hull_Slice(x_MSC, slice_MSC);
% %figure, imshow(Hull_Slice_MSC)
% avg_thresh_MSC = 0.1
% Processed_Hull_MSC = Post_Process_Hull(Hull_Slice_MSC, avg_thresh_MSC);
% figure, imshow(Processed_Hull_MSC)
% title_MSC = [ File_MSC, Data_Set, num2str(slice_MSC),'.png' ];
% %imwrite(Hull_Slice_MSC , title_MSC, 'png', 'bitdepth', 2 )

% FILE_SM = [File_SM, '.txt'];
% %FILE_SM = [File_SM, Data_Set, '.txt'];
% x_SM = importdata(FILE_SM);
% slice_SM = 31;
% Hull_Slice_SM = Return_Hull_Slice(x_SM, slice_SM);
% %figure, imshow(Hull_Slice_SM)
% avg_thresh_SM = 0.9
% Processed_Hull_SM = Post_Process_Hull(Hull_Slice_SM, avg_thresh_SM);
% figure, imshow(Processed_Hull_SM)
% title_SM = [ File_SM, Data_Set, num2str(slice_SM),'.png' ];
% imwrite(Hull_Slice_SM , title_SM, 'png', 'bitdepth', 2 )

%% Post Processed Hull Comparisons
close all;

Phantom = importdata('NeoZdim200.txt');
%Phantom = mat2gray(Phantom);
figure, imshow(Phantom);
Phantom_Dimensions = size(Phantom);
total_hull_voxels = 0;
for i=1:Phantom_Dimensions(1)
    for j=1:Phantom_Dimensions(2)
        if( Phantom(i,j) > 0 )
            total_hull_voxels = total_hull_voxels + 1;
        end
    end
    
end
total_hull_voxels

%%
close all;
 Data_Set = '(sim)';
% Data_Set = '(noisy)';
% Data_Set = '(rat)';
% Data_Set = '(ped)';

File_FBP = 'x_FBP';
File_SC = 'x_MSC';
File_MSC = 'MSC_image_';
File_SM = 'x_sm';
File_BC = 'bin_carve';
% FILE_FBP = [File_FBP, Data_Set, '.txt'];
% x_FBP = importdata(FILE_FBP);
% slice_FBP = 2;
% Hull_Slice_FBP = Return_Hull_Slice(x_FBP, slice_FBP);
% figure, imshow(Hull_Slice_FBP)
% title_FBP = [ File_FBP, Data_Set, num2str(slice_FBP),'.png' ];
% imwrite(Hull_Slice_FBP , title_FBP, 'png', 'bitdepth', 2 )

% FILE_BC = [File_BC, Data_Set, '.txt'];
% x_BC = importdata(FILE_BC);
% slice_BC = 7;
% Hull_Slice_BC = Return_Hull_Slice(x_BC, slice_BC);
% figure, imshow(Hull_Slice_BC)
% %figure, imshow(Hull_Slice_BC)
% avg_thresh_BC = 0.4
% [Processed_Hull_BC, filtered_Hull_BC] = Post_Process_Hull(Hull_Slice_BC, avg_thresh_BC);
% figure, imshow(filtered_Hull_BC)
% figure, imshow(Processed_Hull_BC)
% title_orig_BC = [ File_BC, Data_Set,'orig', num2str(slice_BC),'.png' ];
% title_filt_BC = [ File_BC, Data_Set,'filt', num2str(slice_BC),'.png' ];
% title_final_BC = [ File_BC, Data_Set,'final', num2str(slice_BC),'.png' ];
% imwrite(Hull_Slice_BC , title_orig_BC, 'png', 'bitdepth', 2 )
% imwrite(filtered_Hull_BC , title_filt_BC, 'png', 'bitdepth', 2 )
% imwrite(Processed_Hull_BC , title_final_BC, 'png', 'bitdepth', 2 )

slice_MSC = 31;
FILE_MSC = [File_MSC, num2str(slice_MSC),'.txt'];
x_MSC = importdata(FILE_MSC);
x_MSC = mat2gray(x_MSC);
figure, imshow(x_MSC)
x_MSC_fill = watershed(x_MSC, 4);
figure, imshow(x_MSC_fill)
%avg_thresh_MSC = 0.1
%Processed_Hull_MSC = Post_Process_Hull(Hull_Slice_MSC, avg_thresh_MSC);
%figure, imshow(Processed_Hull_MSC)
%title_MSC = [ File_MSC, Data_Set, num2str(slice_MSC),'.png' ];
%imwrite(Hull_Slice_MSC , title_MSC, 'png', 'bitdepth', 2 )

% FILE_SM = [File_SM, '.txt'];
% %FILE_SM = [File_SM, Data_Set, '.txt'];
% x_SM = importdata(FILE_SM);
% slice_SM = 31;
% Hull_Slice_SM = Return_Hull_Slice(x_SM, slice_SM);
% %figure, imshow(Hull_Slice_SM)
% avg_thresh_SM = 0.9;

% Processed_Hull_SM = Post_Process_Hull(Hull_Slice_SM, avg_thresh_SM);
% figure, imshow(Processed_Hull_SM)
% title_SM = [ File_SM, Data_Set, num2str(slice_SM),'.png' ];
% imwrite(Hull_Slice_SM , title_SM, 'png', 'bitdepth', 2 )

%%
close all;
Phantom = importdata('NeoZdim200.txt');
%Phantom = mat2gray(Phantom);
Phantom_Dimensions = size(Phantom);
figure, imshow(Phantom);

slice_MSC = 5;
FILE_MSC = [File_MSC, num2str(slice_MSC),'.txt'];
x_MSC = importdata(FILE_MSC);
x_MSC = mat2gray(x_MSC);
figure, imshow(x_MSC);
%2
figure, imshow(x_MSC)
center = Phantom_Dimensions(1)/2 
radius = Phantom_Dimensions(1)/2
[x,y] = meshgrid(1:Phantom_Dimensions(1)); 
bw = sqrt((x-center).^2 + (y-center).^2) <= radius - 2; 

% High-Frequency Emphasis Filtering
x_MSC = Iter_Filter2(x_MSC, 'unsharp', 0.9);
figure, imshow(x_MSC);
x_MSC = histeq(x_MSC);
figure, imshow(x_MSC);
level = graythresh(x_MSC);
x_MSC = im2bw(x_MSC,level);
figure, imshow(x_MSC);
x_MSC = Iter_Filter2(x_MSC, 'average',11, 'unsharp', 0.9);
figure, imshow(x_MSC);
x_MSC = bwmorph(x_MSC, 'erode');
figure, imshow(x_MSC);
x_MSC = ~x_MSC & bw;
figure, imshow(x_MSC);
% % figure, imshow(~bw)
% % L = watershed(bw)
% % figure, imshow(L)




%Compute the distance transform of the complement of the binary image.
%D = bwdist(x_MSC);
%figure, imshow(D,[],'InitialMagnification','fit')
%title('Distance transform of ~bw')


%Complement the distance transform, and force pixels that don't belong to the objects to be at -Inf.
%D = -D;
%D(~x_MSC) = -Inf;


%Compute the watershed transform and display the resulting label matrix as an RGB images. 
%L = watershed(x_MSC5);
%figure, imshow(L)
% rgb = label2rgb(L,'jet',[.5 .5 .5]);
% %figure, imshow(rgb,'InitialMagnification','fit')
% title('Watershed transform of D')

%%

center1 = -10;
center2 = -center1;
dist = sqrt(2*(2*center1)^2);
radius = dist/2 * 1.4;
lims = [floor(center1-1.2*radius) ceil(center2+1.2*radius)];
[x,y] = meshgrid(lims(1):lims(2));
bw1 = sqrt((x-center1).^2 + (y-center1).^2) <= radius;
bw2 = sqrt((x-center2).^2 + (y-center2).^2) <= radius;
bw = bw1 | bw2;
figure, imshow(bw,'InitialMagnification','fit'), title('bw')



%Compute the distance transform of the complement of the binary image.
D = bwdist(~bw);
figure, imshow(D,[],'InitialMagnification','fit')
title('Distance transform of ~bw')


%Complement the distance transform, and force pixels that don't belong to the objects to be at -Inf.
D = -D;
D(~bw) = -Inf;
figure, imshow(D,[],'InitialMagnification','fit')

%Compute the watershed transform and display the resulting label matrix as an RGB images. 
L = watershed(D);
rgb = label2rgb(L,'jet',[.5 .5 .5]);
figure, imshow(rgb,'InitialMagnification','fit')
title('Watershed transform of D')