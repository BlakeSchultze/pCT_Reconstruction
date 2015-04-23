% Read text file, perform Radon transform, and produce proton history
% data in the format expected by reconstruction code
close all;

I = importdata('NEOtrue.txt');
I = reshape(I, 160, 200);
I = imrotate(I, 90);
I_temp = zeros(200,200);
I_temp(:,21:180) = I;
I = I_temp;
figure, imshow(I)
angles = 0:0.25:359.5;
[R, ts] = radon(I, angles);
I_size = size(I)
R_size = size(R)
%sinogram = imreshape(R, 
figure, imshow(R)