function [ missing, extra, varargout ] = Phantom_Hull_Comparison( Phantom, Hull_Filename, slice, post_processing, varargin )
    Hull = importdata(Hull_Filename);
    [rows, cols] = size(Hull);
    hull_min = min(min(Hull));
    hull_max = max(max(Hull));
    threshold = (hull_max + hull_min)/2;
    averaging_threshold = 0.4;
    if( nargin == 4 )
        Manual_Resize_Threshold = false;
    else
        Manual_Resize_Threshold = true;
        if(varargin{1} == 'both')
            resize_threshold = varargin{2};
            averaging_threshold = varargin{3};
        elseif( varargin{1} == 'resize' )
            resize_threshold = varargin{2};
        else
            averaging_threshold = varargin{1};
        end
    end
    Hull_Slice = Return_Hull_Slice(Hull, slice);
    Phantom_Dimensions = size(Phantom);
    if( cols~= Phantom_Dimensions(2) )
        if( Manual_Resize_Threshold == true)
            [Hull_Resized, hull_min, hull_max, threshold] = Hull_Resize( Hull_Slice, Phantom_Dimensions(2), resize_threshold);
        else
            [Hull_Resized, hull_min, hull_max, threshold] = Hull_Resize( Hull_Slice, Phantom_Dimensions(2));
        end
        Hull_Comparing = Hull_Resized;
    else
        Hull_Comparing = Hull_Slice;
    end
    if( post_processing == true )
        [ Processed_Hull, Filtered_Hull ] = Post_Process_Hull( Hull_Comparing, averaging_threshold );
        Hull_Comparing = Processed_Hull;
        if(Manual_Resize_Threshold == false)
            varargout{2} = Filtered_Hull; 
        else
            varargout{5} = Filtered_Hull; 
        end
    end
    [missing, extra] = Hull_Accuracy(Phantom, Hull_Comparing);
    varargout{1} = Hull_Comparing;
    varargout{2} = hull_min;
    varargout{3} = hull_max;
    varargout{4} = threshold;
    %if(nargout == 5)
    
end

