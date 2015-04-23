%% Generate shifted/rotated strip correspondence
close all;

t_strips = 100;
t_shift = 15;
t_angle = 30*pi/180;
t_rot_center = 24;

v_strips = 100;
v_shift = 5;
v_angle = 45*pi/180;
v_rot_center = 115;

t_1_v_1_angle = 135*pi/180;

t_1 = 1:t_strips;
v_1 = 1:v_strips;
t_2 = 1:t_strips;
v_2 = 1:v_strips;

% Generate shifted+rotated t_2 strip nummbers
v_2_observed = zeros(length(t_1), length(v_1));
t_2_observed = zeros(length(v_1), length(t_1));
for i = 1:length(v_1)       % i = v_1 value
    for j = 1:length(t_1)   % j = t_1 value
        t_2_observed(i, j) = t_1(j) + t_shift + ( v_1(i) - v_rot_center ) * tan(t_angle);
        ( v_1(i) - v_rot_center ) * tan(t_angle);
    end
end

% Generate shifted+rotated v_2 strip numbers
for i = 1:length(t_1)       % i = t_1 value
    for j = 1:length(v_1)   % j = v_1 value
        v_2_observed(i, j) = v_1(j) + v_shift + ( t_1(i) - t_rot_center ) * tan(v_angle);
    end
end

% Set fitting options
options = fitoptions('Method','LinearLeastSquares');

% Determine angle and intercept between t planes
t_2_observed_rounded = round(t_2_observed);
t_fit_angles = zeros(1, length(t_1));
t_fit_bs = zeros(1, length(t_1));
for t=1:length(t_1)
    t_fit = fit(v_1', t_2_observed_rounded(:, t), 'poly1', options );
    t_fit_angles(t) = t_fit.p1;
    t_fit_bs(t) = t_fit.p2 - t_1(t);
end
mean_t_fit_radians = mean(t_fit_angles)
mean_t_fit_angle = atand(mean_t_fit_radians)
mean_t_fit_b = mean(t_fit_bs)

t_diff = t_2_observed - t_1' * ones(1, length(t_1) );
figure, plot(v_1, t_diff, v_2_observed, t_diff)
t_1_value = 11;
v_1_value = 1;

p1s = zeros(1,length(v_1));
for v=1:length(v_1)
    t_diff_fit = fit(v_1', t_diff(v, :)', 'poly1', options );
    p1s(i) = t_diff_fit.p2;
end
figure, plot(t_diff(v, :))
figure, plot(p1s)

figure, plot(v_1, t_diff(t_1_value, :),'x'), hold on;
plot( v_1, t_diff_fit.p1*t_1 + t_diff_fit.p2, 'g')
%v_angle_fit = atand( v_fit.p1)

t_diff_min_fit = fit(v_1', min(t_diff)', 'poly1', options )
%t_diff_min_fit = fit(v_1', t_diff(:,9), 'poly1', options )
figure, plot(min(t_diff))
vc =( mean_t_fit_b - t_diff_min_fit.p2)/(1 - tan(t_angle))
ts = vc + t_diff_min_fit.p2
a = min(abs(t_diff));
[r,i] = min(a);
%[r2,i2]=min(min(abs(t_diff)));
vc2 =( mean_t_fit_b - i*sign(t_diff(i*length(t_1))))/(1 - tan(t_angle))
ts2 = vc2 + i*sign(t_diff(i*length(t_1)))
%vc2 =( mean_t_fit_b2 - i*sign(t_diff(i*length(t_1))))/(1 - tan(t_angle))

t_fit_angles2 = zeros(1, length(t_1));
t_fit_bs2 = zeros(1, length(t_1));
for t=1:length(t_1)
    t_fit2 = fit(v_2_observed(:,t), t_2_observed_rounded(:, t), 'poly1', options );
    t_fit_angles2(t) = t_fit2.p1;
    t_fit_bs2(t) = t_fit2.p2 - t_1(t);
end
mean_t_fit_radians2 = mean(t_fit_angles2)
mean_t_fit_angle2 = atand(mean_t_fit_radians2)
mean_t_fit_b2 = mean(t_fit_bs2)

t_b_theory = t_shift - mean_t_fit_radians * v_rot_center

t_2_observed_diag = diag(t_2_observed);
mean(t_2_observed_diag - t_1');
% Determine angle and intercept between v planes
v_2_observed_rounded = round(v_2_observed);
v_fit_angles = zeros(1, length(v_1));
v_fit_bs = zeros(1, length(v_1));
for v=1:length(v_1)
    v_fit = fit(t_1', v_2_observed_rounded(:, v), 'poly1', options );
    v_fit_angles(v) = v_fit.p1;
    v_fit_bs(v) = v_fit.p2 - v_1(v);
end
mean_v_fit_radians = mean(v_fit_angles)
mean_v_fit_angle = atand(mean_v_fit_radians)
mean_v_fit_b = mean(v_fit_bs)
v_b_theory = v_shift - mean_v_fit_radians * t_rot_center

%a = (mean_t_fit_b2 - mean_t_fit_b ) / mean_t_fit_radians2


v_diff = v_2_observed - v_1' * ones(1, length(v_1) );
%figure, plot(v_diff)
v_1_value = 10;
t_1_value = 1;

% v_diff_fit = fit(t_1', v_diff(t_1_value, :)', 'poly1', options )
% figure, plot(t_1, v_diff(t_1_value, :),'x'), hold on;
% plot( t_1, v_diff_fit.p1*v_1 + v_diff_fit.p2, 'g')
% %v_angle_fit = atand( v_fit.p1)

% v_2_calc = v_1(1) + v_diff_fit.p1*v_1(1) +  v_diff_fit.p2
% t_2_observed(1, :)

% [r, i] = min(abs(mean(t_diff(:,t_1_value))));
% i = i * sign(t_diff(i, t_1_value));
% v_c = ( b_t_fit - i) / (1 - t_fit.p1)
% Dt_12 = (b_t_fit - i * t_fit.p1)/(1 - t_fit.p1)
% i
% 
% p1s = zeros(1, length(v_1));
% p2s = zeros(1, length(v_1));
% for v = 1:length(v_1)
%     %t_fit2 = fit(v_1', t_2_observed(v, :)', 'poly1', options );
%     t_fit2 = fit(t_1', t_2_observed(:,v), 'poly1', options );
%     %t_fit2 = fit(v_1', t_diff(:, v), 'poly1', options );
%     p1s(v) = t_fit2.p1;
%     p2s(v) = t_fit2.p2;
% end
% p1s
% p2s
% t_fit = fit(v_2_observed(:, t_1_value), t_2_observed(:, t_1_value), 'poly1', options )
% %figure, plot(v_1, t_2_observed(:, t_1_value),'x'), hold on;
% %plot( v_1,t_fit.p1*t_1 + t_fit.p2, 'g')
% t_angle_fit = atand( t_fit.p1)
% b_t_fit = t_fit.p2 - t_1(t_1_value)
% m_t = t_fit.p1;
% b = zeros(1, length(v_1));
% for i = 1:length(v_1)
%     b_fit = fit(t_1', t_diff(v_1(i),:)', 'poly1', options );
%     b(i) = b_fit.p2;
% end
% 'b should be', t_shift - v_rot_center * t_fit.p1
% Dt_fit = fit(v_1', b', 'poly1', options )
%Dt_fit = fit(t_1', t_diff(v_1_value,:)', 'poly1', options )
%Dt_12 = 
%Dt_fit = fit(t_1', t_diff(:, t_1_value), 'poly1', options )
% [optimizer, metric]  = imregconfig('multimodal');
% movingRegistered = imregister(t_1'*ones(1,length(t_1)), t_2_observed(1,:)'*ones(1,length(t_1)),'rigid',optimizer, metric, 'DisplayOptimization', 1);
% T = maketform('affine',t);
% t_2_out = imtransform(t_2_observed(1,:),T)
% figure, imshow(t_2_out)
% t_2_out = imtransform(t_1,T)
% figure, imshow(t_2_out)

% im_t1 = zeros(length(t_1), length(t_1));
% im_t2 = zeros(length(t_1), length(t_1));
% 
% for i = 1:length(t_1)
%    im_t1(t_1(i), round(t_2_observed(1,i))) = i;
% end
% 
% a1 = (mean(t_diff))
% a2 = mean(a)
% a3 = find(a1>0, 1, 'first')
% a4 = a3* t_fit.p1
% (v_rot_center - t_shift* t_fit.p1)
% %[r,c,v] = find(t_diff==a)
% b = a2 - t_fit.p1 * (v_rot_center)/length(t_1)
% b = a2 - t_fit.p1 /2

%% Use WEPL measurements of rod to determine strip #s for t = 0 
close all;

WEPL = [0 0 0 0 1 2 2 4 5 6 6 8 7 9 8 9 7 8 6 6 5 4 4 3 1 1 0 0]
t = 0: length(WEPL) - 1;
t_COM = WEPL * t'/sum(WEPL)
figure, plot(t, WEPL)
WEPL(floor(t_COM)), WEPL(ceil(t_COM))

y = [0.1 0.9 2.1 3.0 3.95 4.9 6.1 7.05]
x = 0:length(y) - 1;
options = fitoptions('Method','LinearLeastSquares');
figure, plot(x,y), hold on;
fitobject = fit(x', y', 'poly1', options )
plot( x,fitobject.p1*x + fitobject.p2)