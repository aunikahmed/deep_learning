function [ output_args ] = getAccuracy(wCell,bCell,unitsPerLayer, images, labels )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
%% init phase
    numLayer = size(unitsPerLayer,2);
    outDim = unitsPerLayer(end);
    m = size(labels,2);
%     y =  zeros(outDim,m);
%     I = sub2ind(size(y),labels,1:m);
%     y(I) = 1;
%        
    aCell = cell(1,numLayer);
    zCell = cell(1,numLayer);
    calculatedLabel = zeros(size(labels));
%% feed forword pass
   for imageNum = 1 : m 

        %aCell{1,1} = reshape(images(:,:,imageNum),[],1);
        aCell{1,1} = images(:,imageNum);
        for L = 2:numLayer
            w = wCell{1,L};
            a = aCell{1,L-1};
            b = bCell{1,L};

            z = w * a + b;
            zCell{1,L} = z;
            aCell{1,L} = f(z);
        end

       % disp(aCell{1,numLayer});
        out = aCell{1,numLayer};
        [val, ind ] = max(out);
        calculatedLabel(1,imageNum) = ind;
        
   end
    output_args = sum(calculatedLabel == labels) * 100/ m;
    
    
    function a = f(z)
        a = 1./(1+ exp(-z)); 
     end
 
end

