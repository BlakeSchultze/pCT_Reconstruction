close all;

% rows = 200;
% cols = 160;
% 
% Phantom = importdata('NeoX_true.txt');
% Phantom = reshape(Phantom,cols,rows)';
% Phantom = mat2gray(Phantom);
% figure, imshow(Phantom)
% Phantom_Dimensions = size(Phantom)
% dlmwrite('NEO_float(200,160).txt', Phantom, 'delimiter', ' ', ...
%          'precision', 6);
% %type NEO_float.txt;
% 
% Phantom = importdata('NeoZdim200.txt');
% Phantom_Dimensions = size(Phantom)

rows = 200;
cols = 200;

PhantomFloat = importdata('NEO_float(200,160).txt');
RSP_values = 0;
num_RSP_values = 1;
new_value = true;

PhantomFloat_Dimensions = size(PhantomFloat)
for row = 1:PhantomFloat_Dimensions(1)
	for col = 1:PhantomFloat_Dimensions(2)
        RSP = PhantomFloat(row,col);
        ind = find(RSP_values == RSP );
        if( isempty(ind) )
             RSP_values = [RSP_values RSP];
             num_RSP_values = num_RSP_values + 1;
        end
	end
end
RSP_values
% Phantom = importdata('NEO_float(200,200).png');
% Phantom = imresize(Phantom,[cols,rows]);
% Phantom = mat2gray(Phantom);
% Phantom = Phantom(:,:,1);
% figure, imshow(Phantom)
% Phantom_Dimensions = size(Phantom)
% dlmwrite('NEO_float(200,200).txt', Phantom, 'delimiter', ' ', 'precision', 6);
% %type NEO_float.txt;