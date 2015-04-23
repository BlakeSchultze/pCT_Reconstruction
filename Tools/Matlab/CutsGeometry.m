close all;

% x = 3;
% y = 4;
% z = 5;
% 
% d_xz = sqrt(3^2+5^2)
% 
% 
% theta_xy = atand(4/3)
% theta_xz = atand(5/3)
% theta_xyz = atand(5/d_xz)
% % 
% % sqrt(abs(theta_xy^2 - theta_xyz^2))
% 
% theta_1 = atan(4/3)
% theta_2 = atan(5/3)
% theta_3 = atan(5/d_xz)
% %theta_4 = atan(  4^2 + 3^2)
% theta_5 = atan(d_xz/3)
% sqrt( atan(4/3) + atan(5/3))
% atan(3^2+4^2)
% atan(3^2+5^2)
% atan(4^2+5^2)
% atan(3^2+4^2+5^2)

filename = 'FBP_image_h_15.txt';
%image = DisplayTextImage( filename );
%images = {image}
filename2 = 'x_FBP.txt';
%image2 = DisplayTextImage( filename );
%images = [images, {image2}]
%images = DisplayTextImages( filename, filename2 )
%images = Text2PNGImages(  filename, filename2 )
% for i = 1:size(images, 2)
%     figure, imshow( images{i});
% end

name = filename;
names = {name};
name = filename2;
names = [names,{name}]
names{1}
images = ImportTextImages( names )
DisplayTextImages( images )
%images = Text2PNGImages(  names )