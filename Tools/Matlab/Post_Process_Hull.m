function [ Processed_Hull, varargout ] = Post_Process_Hull( Hull, varargin )
    if(nargin== 2)
        averaging_threshold = varargin{1};
    else
        averaging_threshold = 0.4;
    end
    [rows, cols] = size(Hull);
    %figure, imshow(Hull);
    % Average filtered bin carving image
    h = fspecial('average', 7);
    Filtered_Hull = imfilter( Hull, h);
    %figure, imshow(Filtered_Hull);
    %Processed_Hull = zeros(rows, cols);
    % Do the thresholding
    for i = 1 : rows
        for j = 1 : cols
            if(Filtered_Hull(i,j) > averaging_threshold )
                Processed_Hull(i,j) = 1;
            else
                Processed_Hull(i,j) = 0;
            end
        end
    end  
    %figure, imshow(Processed_Hull);
    varargout{1} = Filtered_Hull;
end

