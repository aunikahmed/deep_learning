function [ weightCell, biasCell ] = mnn( images, labels, unitsPerLayer, weight, bias )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
%% init phase
    numLayer = size(unitsPerLayer,2);
    outDim = unitsPerLayer(end);
    m = size(labels,2);
    y =  zeros(outDim,m);
    I = sub2ind(size(y),labels,1:m);
    y(I) = 1;
       
    [wCell, bCell] = stackWeightIncell(weight,bias,unitsPerLayer);
    
    aCell = cell(1,numLayer);
    zCell = cell(1,numLayer);
    
    
   
    
   
    
   alpha = .01;
   for epoc = 1 : 1
       
       [deltaWCell, deltaBCell] = initDelta; % initializing dW and dB
       
       for imageNum = 1 : m
    %% feed forword pass
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
    %% backpropagation     
            deltaCell = cell(1,numLayer);
            aOut = aCell{1,numLayer};
            fPrime = aOut .* (1 -  aOut);
            deltaCell{1,numLayer} = -( y(:,imageNum) - out ) .* fPrime;


            for L = numLayer -1 : -1 : 2
               w = wCell{1,L + 1};
               deltaPrev = deltaCell{1, L + 1};
               a =  aCell{1, L};
               fPrime = a .* (1 - a);
               delta = (w' * deltaPrev) .* fPrime;
               deltaCell{1,L} = delta;
            end



            for L = 2 : numLayer
                delta = deltaCell{1,L};
                a = aCell{1,L-1};
                
                deltaWCell{1,L} = deltaWCell{1,L} + delta * a';
                deltaBCell{1,L} = deltaBCell{1,L} + delta;
            end


            %disp(imageNum);
        end


         for L = 2 : numLayer
            dW = deltaWCell{1,L};
            dB = deltaBCell{1,L};


             w = wCell{1, L};
             b = bCell{1,L};
             w = w - alpha * (dW/m);
             b = b - alpha * (dB/m);
             wCell{1,L} = w;
             bCell{1,L} = b;
         end
        fprintf('epoch %d\n',epoc);
   end
    
    
    
    
    
 %% utility function 
    function[deltaWCell, deltaBCell] = initDelta 
        deltaWCell = cell(size(wCell));
        deltaBCell = cell(size(bCell));

        for l = 2:numLayer
            deltaWCell{1,l} = zeros(size(wCell{1,l}));
            deltaBCell{1,l} = zeros(size(bCell{1,l}));
        end
    end

    function a = f(z)
        a = 1./(1+ exp(-z)); 
    end
    weightCell = wCell;
    biasCell = bCell;
end

