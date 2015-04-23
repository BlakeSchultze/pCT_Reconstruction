function [ Missing, Extra ] = Hull_Accuracy( Hull_1, Hull_2 )
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
    [rows, cols] = size(Hull_1);
    Missing = 0;
    Extra = 0;
    for i = 1:rows
        for j = 1:cols
            Hull_Diff = Hull_2(i,j) - Hull_1(i,j);
           if( Hull_Diff == 1 )
                Extra = Extra + 1;
           end
           if( Hull_Diff == -1 )
                Missing = Missing + 1;
           end
        end
    end
end

