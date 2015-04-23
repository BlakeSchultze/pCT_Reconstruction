

% Read text file, perform Radon transform, and produce proton history
% data in the format expected by reconstruction code
close all;

I = importdata('SM_image_4.txt');
I = mat2gray(I);
figure, imshow(I);
h = fspecial('average')
I2 = imfilter(I, h);
I2 = imfilter(I2, h);
% for i = 1:length(I)
%     for j = 1:length(I)
%        if( (i - 1 -(length(I)/2))^2 + (j- 1 -(length(I)/2))^2 > (length(I)/2)^2 )
%            I(i,j) = 0;
%        end
%     end
% end
% for i = 1:length(I)
%     for j = 1:length(I)
%        if( (i -(length(I)/2))^2 + (j-(length(I)/2))^2 > (length(I)/2)^2 )
%            I(i,j) = 0;
%        end
%     end
% end
% figure, imshow(I);
% [BW,thresh] = edge(I, 'sobel');
% thresh
% figure, imshow(BW);
% [BW,thresh] = edge(I,'prewitt');
% thresh
% figure, imshow(BW);
% [BW,thresh] = edge(I,'roberts');
% thresh
% figure, imshow(BW);
% [BW,thresh] = edge(I,'log');
% thresh
% figure, imshow(BW);
% [BW,thresh] = edge(I,'zerocross');
% thresh
% figure, imshow(BW);
% %[BW,thresh] = edge(I,'canny');
figure, imshow(I2);
 BW = edge(I2,'canny', .15);
figure, imshow(BW);
% BWfill = imfill(BW,'holes');
% figure, imshow(BWfill);
% %I(length(I)/2,length(I)/2)
% avgval = mean2(I)
% B = imregionalmin(I);
% figure, imshow(B)
% h = fspecial('laplacian')
% I2 = imfilter(I, h);
% figure, imshow(I2)
% BW = edge(I2,'canny',0.01);
% figure, imshow(BW);
% BWfill = imfill(BW);
% figure, imshow(BWfill);
% h = fspecial('average')
% I2 = imfilter(BWfill, h);
% I2 = imfilter(I2, h);
% figure, imshow(I2)
% for i = 1:length(I)
%     for j = 1:length(I)
%        if(I(i,j) < 0.04 & I(i,j) ~= 0 )
%        %if( I(i,j) < 0.04 )
%            I(i,j) = 1;
%        elseif( (i-(length(I)/2))^2 + (j-(length(I)/2))^2 > (length(I)/2)^2 )
%            I(i,j) = 0;
%        else
%            I(i,j) = 0;
%        end
%     end
% end
% figure, imshow(I);
% I = reshape(I, 160, 200);
% I = imrotate(I, 90);
% I_temp = zeros(200,200);
% I_temp(:,21:180) = I;
% I = I_temp;
% figure, imshow(I)
% angles = 0:0.25:359.5;
% [R, ts] = radon(I, angles);
% I_size = size(I)
% R_size = size(R)
% %sinogram = imreshape(R, 
% figure, imshow(R)