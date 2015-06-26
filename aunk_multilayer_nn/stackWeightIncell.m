function [weightCell, biasCell] = stackWeightInCell(weight,bias, unitsPerLayer )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    numLayer = size(unitsPerLayer,2);
    wCell = cell(1, numLayer);
    bCell = cell(1, numLayer);
    indWS = 1;
    indBS = 1;
    for L = 2:numLayer
        unitsInCurrentLayer = unitsPerLayer(L);
        unitsInPrevLayer = unitsPerLayer(L-1);
        
        indWE = indWS + unitsInCurrentLayer * unitsInPrevLayer ;
        indBE = indBS + unitsInCurrentLayer;
        W = reshape(weight(indWS:indWE-1),unitsInCurrentLayer,unitsInPrevLayer);
        b = bias(indBS:indBE-1);
        wCell{L} = W;
        bCell{L} = b;
        
    end

    weightCell = wCell;
    biasCell = bCell;
    


end

