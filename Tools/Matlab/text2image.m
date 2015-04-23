% Read text files from pCT reconstruction and produce .png images
% of each slice
close all;

% I1 = importdata('x_FBP.txt');
% I1 = mat2gray(I1);
% I2 = importdata('x_FBP(1).txt');
% I1 = mat2gray(I2);
% size(I1)
% size(I2)
% I3 = I2-I1;
% figure, imshow(I1)
% figure, imshow(I2)
% figure, imshow(I3)
% for i=1:length(I3)
%    if( I3(i) == 1)
%       i 
%    end
% end
% Imported image postfixes
% rat, sim, ped


 filename_tag = 'rat';
 sim_image = 4;
 filename = ['bin_carve(',filename_tag,')']
 I = importdata([filename,'.txt']);
 [im_rows, cols] = size(I)
  rows = cols
slices = im_rows/cols

FBP_filename = ['x_FBP(',filename_tag,')'];
I_FBP = importdata([FBP_filename,'.txt']);
[FBP_im_rows, FBP_cols] = size(I_FBP)
FBP_rows = FBP_cols
FBP_slices = FBP_im_rows/FBP_cols

SC_filename = ['x_sc(',filename_tag,')'];
I_SC = importdata([SC_filename,'.txt']);
[SC_im_rows, SC_cols] = size(I_SC)
SC_rows = SC_cols
SC_slices = SC_im_rows/SC_cols

MSC_filename = ['x_msc(',filename_tag,')'];
I_MSC = importdata([MSC_filename,'.txt']);
[MSC_im_rows, MSC_cols] = size(I_MSC)
MSC_rows = MSC_cols
MSC_slices = MSC_im_rows/MSC_cols

SM_filename = ['x_sm(',filename_tag,')'];
I_SM = importdata([SM_filename,'.txt']);
[SM_im_rows, SM_cols] = size(I_SM)
SM_rows = SM_cols
SM_slices = SM_im_rows/SM_cols


 averaging_threshold = 0.4;
% for i = 7 :7  %sim 5,7
% for i = 7 :7  %ped 5,7
%for i = sim_image :sim_image  %sim
for i=7:7
    % Original image
    first = 1 + rows * ( i - 1 );
    last = rows * i;
    figure, imshow( I(first:last, :) );
    title = [ filename, num2str(i),'.png' ];
    %imwrite( I( first:last, : ), title, 'png', 'bitdepth', 2 )
    
    
    % Average filtered bin carving image
    h = fspecial('average',5);
    Ifilt = imfilter( I(first:last, :), h, 'replicate');
    figure, imshow( Ifilt );
    %imwrite( Ifilt, 'bin_carve_avg7(rat).png', 'png', 'bitdepth', 2 )
    % Do the thresholding
    for j = 1 : rows
        for k = 1 : cols
            if(Ifilt(j,k) > averaging_threshold )
                Ifilt(j,k) = 1;
            else
                Ifilt(j,k) = 0;
            end
        end
    end  
    figure, imshow(Ifilt)
    title_filtered = [ filename,num2str(i),'.png' ];
    %imwrite( Ifilt, title_filtered, 'png', 'bitdepth', 2 )
end
%for i = 1: FBP_slices
for i = sim_image: sim_image %sim
    % FBP image 
    first_FBP = 1 + FBP_cols * ( FBP_slices - i - 1 );
    last_FBP = FBP_cols * (FBP_slices -i);
    figure, imshow(I_FBP(first_FBP:last_FBP, :))
    title_FBP = [ FBP_filename,num2str(i),'.png' ];
    imwrite( I_FBP(first_FBP:last_FBP, :), title_FBP, 'png', 'bitdepth', 2 )
end

%for i = 2:2 %ped
%for i = 2:2 %rat
for i = sim_image:sim_image %sim
    % SC image 
    first_SC = 1 + SC_cols * ( i - 1 );
    last_SC = SC_cols * i;
    figure, imshow(I_SC(first_SC:last_SC, :))
    title_SC = [ SC_filename,num2str(i),'.png' ];
    imwrite( I_SC(first_SC:last_SC, :), title_SC, 'png', 'bitdepth', 2 )
end

for i = sim_image:sim_image %ped
%for i = 7:7 %rat
%for i = sim_image:sim_image %sim
    % MSC image 
    first_MSC = 1 + MSC_cols * ( i - 1 );
    last_MSC = MSC_cols * i;
    figure, imshow(I_MSC(first_MSC:last_MSC, :))
    title_MSC = [ MSC_filename,num2str(i),'.png' ];
    imwrite( I_MSC(first_MSC:last_MSC, :), title_MSC, 'png', 'bitdepth', 2 )
end
%for i=17:17 %ped
%for i = 7:7 %rat
for i = sim_image:sim_image %sim
    % SM image 
    first_SM = 1 + SM_cols * ( i - 1 );
    last_SM = SM_cols * i;
    figure, imshow(I_SM(first_SM:last_SM, :))
    title_SM = [ SM_filename,num2str(i),'.png' ];
    imwrite( I_SM(first_SM:last_SM, :), title_SM, 'png', 'bitdepth', 2 )
end