% theta_xy = 30;
% theta_xz = 50;
% Dy_Dx = tand(theta_xy)
% Vx = 1;
% Dz_Dx = tand(theta_xz)
% 
% Dy = Dy_Dx*Vx
% Dz = Dz_Dx*Vx
% L_xy_1 = sqrt(Vx^2+Dy^2)
% L_xy_2 = Vx/cosd(theta_xy)
% L_xy_3 = Vx*sqrt(1+Dy_Dx^2)
% 
% L_xz_1 = sqrt(Vx^2+Dz^2)
% L_xz_2 = Vx/cosd(theta_xz)
% L_xz_3 = Vx*sqrt(1+Dz_Dx^2)
% 
% L_xyz_1 = sqrt(L_xy_1^2+L_xz_1^2)
% L_xyz_2 = Vx*sqrt(2+Dy_Dx^2+Dz_Dx^2)
% 
% L1 = sqrt(Vx^2+Dy^2+Dz^2)
% L2 = sqrt(L_xy_1^2+Dz^2)
% L3 = Vx/sqrt(1+Dy_Dx^2+Dz_Dx^2)
% L4 = Vx^2/sqrt(Vx^2+Dy^2+Dz^2)

% theta = 0:0.001:pi/4;
% integral = 0;
% for i = 1:length(theta)
%    integral = integral + sqrt(1+sin(theta(i))^2); 
% end
% integral
% Davg = (4/pi)*integral

A0	= 7.457E-6
A1	=4.548E-7
A2 = -5.777E-8
A3	= 1.301E-8 
A4	= -9.228E-10
A5	= 2.687E-11 

poly_1_2 = zeros(1, length(u) );
poly_2_3 = zeros(1, length(u) );
poly_3_4 = zeros(1, length(u) );
poly_2_6 = zeros(1, length(u) );
poly_3_12 = zeros(1, length(u) );
u = 0:0.0001:20;
for i = 1:length(u)
   i*0.0001
   poly_1_2(i) = A0*u(i) + A1/2*u(i)^2 + A2/3*u(i)^3 + A3/4*u(i)^4 + A4/5*u(i)^5 + A5/6*u(i)^6;
   poly_2_3(i) = A0/2*u(i)^2 + A1/3*u(i)^3 + A2/4*u(i)^4 + A3/5*u(i)^5 + A4/6*u(i)^6 + A5/7*u(i)^7;
   poly_3_4(i) = A0/3*u(i)^3 + A1/4*u(i)^4 + A2/5*u(i)^5 + A3/6*u(i)^6 + A4/7*u(i)^7 + A5/8*u(i)^8;
   poly_2_6(i) = A0/2*u(i)^2 + A1/6*u(i)^3 + A2/12*u(i)^4 + A3/20*u(i)^5 + A4/30*u(i)^6 + A5/42*u(i)^7;
   poly_3_12(i) = A0/3*u(i)^3 + A1/12*u(i)^4 + A2/30*u(i)^5 + A3/60*u(i)^6 + A4/105*u(i)^7 + A5/168*u(i)^8;
end




