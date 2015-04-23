 close all;
clear all;
clc;

% T = timer('TimerFcn', @mycallback, 'ExecutionMode', 'singleShot')
% @mycallback
% start(T, @mycallback)
tic
 golden = (1+sqrt(5))/2;
bad_nums = [0];
for iteration = 1:1
    for num_elements = 30000:30500
        sequence = zeros(1, num_elements);
        sequence(1) = 1;
        for j = 2:num_elements
            num = sequence(j-1) + round(num_elements/golden);
            sequence(j) = mod(num, num_elements);
        end
        for k = 0:num_elements-1
           ind = find(sequence == k );
           if( size(ind,2) ~= 1 & bad_nums(end) ~= num_elements)
               bad_nums = [bad_nums, num_elements];
           end
        end
    end
end

bad_nums = bad_nums(2:end)
length(bad_nums)
toc
% stop(T)
%timer('TimerFcn',@mycallback)
%timer(@mycallback,'AveragePeriod')
%  dir_data = dir(fullfile([output_directory, output_folder, iterates_basename,'*.txt']))
% 
% % for file = 1:length(av_files)
% %     if(  av_files(1).name == [iterates_basename, '.txt'])
% % end
% start_iteration = 1;
% end_iteration = 0;
% iterations = [];
% for file = 1:length(dir_data)
%     found = strmatch( [iterates_basename, num2str(end_iteration + 1),'.txt'], file_names);
%     if( found )
%         end_iteration = end_iteration + 1;
%         iterations = [iterations, file];
%     end
% end
% end_iteration
% iterations
% file_names = {dir_data.name}
% for 
% strmatch( [iterates_basename, num2str(start_iteration),'.txt'], file_names);
% ind = find(cell2mat(file_names), [iterates_basename, num2str(start_iteration),'.txt'] )
% input_directory = 'C:\Users\Blake\Documents\Visual Studio 2010\Projects\pCT_Reconstruction\Output\input_water_GeantNONUC\';
% text_image = read_pCT_txt_image([input_directory, 'x']);
% display_text_images({text_image}, true);

% R = 10.0;
% r = 0:0.01:R;
% 
% y = zeros(1,length(r));
% z = zeros(1,length(r));
% for i=1:length(y)
%    %y(i) = 0.0001*(10*R^2 - 9*r(i)^2)/(10*r(i)^2);
%    y(i) = 0.0001*(1/(r(i)/sqrt(R+5)+1)^2);
%    %y(i) = (10*R^2 - 9*r(i)^2)/(10*R^2);
%    %z(i) = r(i)^-2;
% end
% figure, plot(r, y);
%figure, plot(r, z);
% columns = 200;
% rows = 200;
% slices = 32;
% 
% column_in = 100;
% row_in = 75;
% slice_in = 12;
% voxel = column_in + row_in * columns + slice_in * columns * rows
% [column, row, slice] = voxel_2_indices(voxel, columns, rows)  
% thetai = 1 * pi/180;
% theta = 0.01:0.01:pi/2-0.01;
% 
% func1p = zeros(1, length(theta));
% func2p= zeros(1, length(theta));
% for i = 1:length(theta)
% C = cos(theta(i));
% S = sin(theta(i));
% B = cos(theta(i)) + sin(theta(i));
% S2 = sin(2*theta(i));
% A = 0.5;
% l = 1;
% D = 0.25;
% 9535892
% 9535929
% 9535939
% 9535949
% 9535986
% 9535992
% 9535021
% func1p(i) = (l/3) * ( (A^2*S2 - 6)/(A*S2 - 2*B) + (A^2*S2)/(2*B) );
% A = A *B;
% D = D * B;
% C = cos(thetai);
% S = sin(thetai);
% B = cos(thetai) + sin(thetai);
% S2 = sin(2*thetai);
% func2p(i) = (l/3) * ( (A^2*S2 - 6*D)/(A*S2 - 2*(D*C+S)) + (A^2*S2)/(2*(D*C + S)) );
% 
% end
% 
% C = cos(thetai);
% S = sin(thetai);
% B = cos(thetai) + sin(thetai);
% S2 = sin(2*thetai);
% figure, plot( theta, func1p );
% figure, plot( theta, func2p );
% func1 = (l/3) * ( (A^2*S2 - 6*D)/(A*S2 - 2*B) + (A^2*S2)/(2*B) )
% func2 = (l/3) * ( (A^2*S2 - 6*D)/(A*S2 - 2*D*C+S) + (A^2*S2)/(2*D*C + S) )
% func3 = (l/3) * (A^3*S2^2 - 12*D*(C+S))/(2*A*S2*(C+S) - 4*D*(C+S)^2)
% 
% B = cos(thetai) + sin(thetai);
% func4 = (l/(6*B*D))*( (A^3*S2^2 - 12*B*D^2)/(A*S2 - 2*B*D) )
% d = 0.08;
% s = .25;
% r = 8.0;
% h = 4;
% x_in = -1.0;
% y_in = 2.1;
% z_in = -3.1;
% 
% x_out = -3.0;
% y_out = 1.0;
% z_out = 1.0;
% % 
% dy_dx = ( y_out - y_in ) / ( x_out - x_in )
% dz_dx = ( z_out - z_in ) / ( x_out - x_in )
% dy_dz = ( y_out - y_in ) / ( z_out - z_in )
% % 
% xy_angle = atan2( y_out - y_in, x_out - x_in )
% xz_angle = atan2( z_out - z_in, x_out - x_in )
% sin(xz_angle)
% cos(xy_angle)
% %double dy_dx = ( y_out - y_in ) / ( x_out - x_in );
% 
% %	x_move_direction = ( cos(xy_angle) >= 0 ) - ( cos(xy_angle) < 0 );
% %	y_move_direction = ( sin(xy_angle) >= 0 ) - ( sin(xy_angle) < 0 );
% %	z_move_direction = ( sin(xz_angle) >= 0 ) - ( sin(xz_angle) < 0 );
% 
% mx = ( cos(xy_angle) >= 0 ) - ( cos(xy_angle) < 0 )
% my = ( sin(xy_angle) >= 0 ) - ( sin(xy_angle) < 0 )
% mz = ( sin(xz_angle) >= 0 ) - ( sin(xz_angle) < 0 )
% if( mx )
%     mz = ( tan( xz_angle ) > 0 )
% else
%     mz = ( tan( xz_angle ) < 0 ) 
% end
%x = -pi/2+0.01:0.01:pi/2-0.01;
%figure, plot(x,tan(x))
% x_move_direction = ( x_out >= x_in ) - ( x_out <= x_in )
% y_move_direction = ( y_out >= y_in ) - ( y_out <= y_in )
% z_move_direction = ( z_out >= z_in ) - ( z_out <= z_in )
% 
% vx = floor( (x_in + r) / d )
% vx_next = vx + x_move_direction
% x_next = vx_next * d - r
% 
% vy = floor( (r - y_in) / d )
% vy_next = vy - y_move_direction
% y_next = r - (vy_next+1) * d
% 
% vz = floor( (h - z_in) / s )
% vz_next = vz - z_move_direction
% z_next = h - (vz_next+1) * s
% 
% x_at_x_edge = x_next
% x_at_y_edge = (1/dy_dx) * ( y_next - y_in ) + x_in
% x_at_z_edge = (1/dz_dx) * ( z_next - z_in ) + x_in
% 
% x_dist_2_x_edge = x_next - x_in
% x_dist_2_y_edge = x_at_y_edge - x_in
% x_dist_2_z_edge = x_at_z_edge - x_in





%y_at_x_edge = dy_dx * ( x_next - x_in) + y_in
%yr = y - y_in
% 
% close all;
% 
% input_directory = 'C:\Users\Blake\Documents\Visual Studio 2010\Projects\pCT_Reconstruction\Output\input_water_GeantNONUC\';
% 
% filename_base = 'x_';
% 
% txt_2_PNG( [filename_base, 'FBP'] );
% 
% txt_2_PNG( [filename_base, 'FBP'], [filename_base, 'FBP2'] );
% a = [1 1 0 1];
% b = [0 1 1];
% c = [0 0 1 1 0];
% e = {a, b, c}
% e = {a, b};
% e = [e, {c}]
% 
% f = []
% f = [f {a}]
% f = [f {b} {c}]
% f2 = [{a} {b}]
% f3 = [{a} {b} {c} {b}]
% g = [{e} {e2} {e3}]
% size(g)
% g{1}{1}
% g{1}{2}
% g{1}{3}
% 
% g{2}{1}
% g{2}{2}
% 
% g{3}{1}
% g{3}{2}
% g{3}{3}
% g{3}{4}

% filename = 'pCTraw_Run_415.out';
% fid = fopen(filename);
% %data = textscan(fid, '%c' )
% A = importdata(filename);
% radians = 0:0.01:2*pi;
% degrees = 0:0.5:360;
% 
% rel_radians = zeros(length(radians), length(radians));
% size(rel_radians);
% for i = 1:length(radians)
%     for j=1:length(radians)
%         diff = radians(j) - radians(i);
%         rel_radians(i,j) = diff;
%     end
%     
% end
% figure, surfc( rel_radians);
% 
% rel_degrees = zeros(length(degrees), length(degrees));
% size(rel_degrees);
% for i = 1:length(degrees)
%     for j=1:length(degrees)
%         diff = degrees(j) - degrees(i);
%         %rel_degrees(i,j) = min( min(diff,diff - 360), max(diff,diff + 360));
%         %rel_degrees(i,j) = min( min(diff,diff - 360), max(diff,diff + 360));
%         if( diff > 180 )
%             rel_degrees(i,j) =  diff - 360;
%         elseif( diff < -180 )
%             rel_degrees(i,j) = diff + 360;
%         else
%             rel_degrees(i,j) = diff;
%         end
%     end
%     
% end
% figure, surfc(degrees,degrees, rel_degrees);
% shading interp
% rotate3d
% degrees2 = -180:0.5:180;
% rel_degrees2 = zeros(length(degrees2), length(degrees2));
% size(rel_degrees2);
% for i = 1:length(degrees2)
%     for j=1:length(degrees2)
%         diff = degrees2(j) - degrees2(i);
%         %rel_degrees(i,j) = min( min(diff,diff - 360), max(diff,diff + 360));
%         %rel_degrees(i,j) = min( min(diff,diff - 360), max(diff,diff + 360));
%         if( diff > 180 )
%             rel_degrees2(i,j) =  diff - 360;
%         elseif( diff < -180 )
%             rel_degrees2(i,j) = diff + 360;
%         else
%             rel_degrees2(i,j) = diff;
%         end
%     end
%     
% end
% figure, surfc(degrees2,degrees2, rel_degrees2);
% shading interp
% rotate3d

% a0 = 1;
% a1 =2;
% a2 =3;
% a3 =4;
% a4 =5;
% a = [a0,a1,a2,a3,a4];
% 
% num_vals = 10;
% x_vals = 1:num_vals;
% x = zeros(num_vals,5);
% 
% W_diags = ones(1, num_vals);
% W = diag(W_diags,0)
% for i = 1:num_vals
%    x(i,1) = 1;
%    x(i,2) = x_vals(i);
%    x(i,3) = x_vals(i)^2;
%    x(i,4) = x_vals(i)^3;
%    x(i,5) = x_vals(i)^4;
% end
% y = zeros(num_vals,1);
% for i = 1:num_vals
%    y(i) = x(i,:)*a';
%     %y(i) = sum(x(i,:));
% end
% y;
% %y = a0 + a1 * x(:,1) + a2*x(:,2).^2 + a3*x(:,3).^3 + a4.*x(:,4).^4
% for i = 1:20
% %b = (x'*x)/(x'*y)
% %b = inv(sum(x*x')).*sum(x*y);
% %b = inv(x'*x)*x'*y
% i
% b = inv(x'*W*x)*x'*W*y
% r = y - x*b
% for j = 1: size(W,1)
%     W(j,j) = 1/r(j)^2;
% end
% r_tot = (b-a')'*(b-a')
% end