function [ missing, extra, Hull_Slice, PC_Hull_Slice ] = Hull_Hull_Comparison( Hull_Filename, PC_Hull_Filename, slice )
    Hull = importdata(Hull_Filename);
    PC_Hull = importdata(PC_Hull_Filename);
    Hull_Slice = Return_Hull_Slice(Hull, slice);
    PC_Hull_Slice = Return_Hull_Slice(PC_Hull, slice);
    [missing, extra] = Hull_Accuracy(Hull_Slice,PC_Hull_Slice);
end


