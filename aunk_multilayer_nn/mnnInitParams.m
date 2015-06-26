function  [weight,bias] = mnnInitParams( unitsPerLayer )
%MNNINITPARAMS Summary of this function goes here
%   Detailed explanation goes here
    numLayer = size(unitsPerLayer,2);
    w = []; 
    b = [];
    e  = .01;
    for  L = 2:numLayer
        unitsInCurrentLayer = unitsPerLayer(L);
        unitsInPrevLayer = unitsPerLayer(L-1);
        
        w_temp = normrnd(0,e,unitsInCurrentLayer * unitsInPrevLayer,1);
        b_temp = normrnd(0,e,unitsInCurrentLayer,1);
        
        w = [w ; w_temp(:)];
        b = [b ; b_temp(:)]; 
    end
    
    weight = w;
    bias = b;


end

