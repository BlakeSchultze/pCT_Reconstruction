function [ Resized_Hull, varargout ] = Hull_Resize( Hull, New_Size, varargin )
% Resizes the hull in cases where the size of the phantom is different
    Resized_Hull = imresize(Hull,[New_Size,New_Size]);
    [rows, cols] = size(Resized_Hull);
    Hull_Min = min(min(Resized_Hull));
    Hull_Max = max(max(Resized_Hull));
    Hull_Range = Hull_Max - Hull_Min;
    if(nargin == 2)
        Threshold = Hull_Max - Hull_Range/2;
    else
        Threshold = varargin{1};
    end
	for i = 1:rows
        for j = 1:cols          
            if( Resized_Hull(i,j) > Threshold )
                Resized_Hull(i,j) = 1;
            else
                Resized_Hull(i,j) = 0;
            end
        end
    end
    if( nargout > 1 )
        varargout{1} = Hull_Min;
        varargout{2} = Hull_Max;
        varargout{3} = Threshold;
    end
end

