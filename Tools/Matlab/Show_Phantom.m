%% Phantom Viewing
close all;

title_end = '_RStP_dicom_phantom_txt125mm';
% for i=1:1
%     title_begin = int2str(i);
%     title = [title_begin title_end '.txt'];
%     Phantom = importdata(title);
%     title = [title_begin title_end '.dat'];
%     fileID = fopen(title)
%     fwrite( title, Phantom );
% end
slice = 76;
title_begin = int2str(slice)
title = [title_begin title_end '.txt']
%  Phantom_ID = fopen( title, 'wt+');
%  fclose('all')
 %Phantom = fread(Phantom_ID );
Phantom = importdata(title);
Phantom(521, 890 )
Phantom(521, 891 )
Phantom(521, 892 )
[xmax, ymax] = size(Phantom);
Phantom = mat2gray(Phantom);
start_vx = 891;
start_vy = 521;
v_width = 0.1875;
v_height = v_width;
%figure, imshow(Phantom);
%hold on;
%line([0 512.9],[0 256.1],'Color','g');
%line([1024 512.0],[1024 256.9],'Color','g');
markerInserter = vision.MarkerInserter('Shape','Circle','BorderColor','white','Size',4,'Fill',true,'FillColor','Custom','CustomFillColor',[0 0 1]);
Entries = importdata(title)
Pts = [891 521];
Phantom = repmat(Phantom,[1 1 3]);
J = step(markerInserter, Phantom, Pts);
x_final = 17.262;
x_min = -96;
v_width = 0.1875;
target_vx = int32(( x_final - x_min ) / v_width  + 1)
partial_vx = ( x_final - x_min ) / v_width  + 1 - target_vx
target_vy = 521;
%remaining_x = v_width * ( target_vx
Phantom = importdata(title);
WED = 0;
L = 0;
j = 0;
for i = 606:891
   WED = WED + Phantom( 521, i ) * v_width;
   L = L + v_width;
   j = j + 1;
end
WED
L
j
% for i = 1:1024
%    J( 521,i ) = 0;
% end
% Image(row,col)
figure, imshow(J);
% slice = 76;
% title_begin = int2str(slice);
% title = [title_begin title_end];
% 
% Phantom = importdata(title);
% Phantom = mat2gray(Phantom);
% figure, imshow(Phantom);
% Phantom_Dimensions = size(Phantom)