%% Phantom Viewing
close all;

markerInserter = vision.MarkerInserter('Shape','Circle','BorderColor','white','Size',4,'Fill',true,'FillColor','Custom','CustomFillColor',[0 0 1]);
Entries = importdata('Entries_0.txt')
Entries = sortrows(Entries, 3)
[rows,cols] = size(Entries);
previous_slice = Entries(1,3);
title_begin = int2str(previous_slice);
title_end = '_RStP_dicom_phantom_txt125mm';
title = [title_begin title_end '.txt'];
Phantom = mat2gray(importdata(title));
%Phantom = mat2gray(Phantom);
Pts = Entries(1,1:2);
for i=2:rows
    slice = Entries(i,3);
    if( slice == previous_slice )
        Pts = [Pts; Entries(i,1:2)];
    else
        Phantom = repmat(Phantom,[1 1 3]);
        J = step(markerInserter, Phantom, Pts);
        figure, imshow(J); 
        title_begin = int2str(slice);
        title = [title_begin title_end '.txt'];
        Phantom = mat2gray(importdata(title));
        Pts = Entries(i,1:2);
    end
    previous_slice = slice;
end
