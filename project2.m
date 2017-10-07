data = csvread('data.csv');
[train,test] = distributeData(data,0.75);

function [training,testing] = distributeData(data1,proportion)
%This function returns two randomly distributed data sets as training and 
%testing based on proportion (that should be training).
    n = size(data1,1);
    split = round(n*proportion);
    seq = randperm(n);
    training = data1(seq(1:split),1:end);
    testing = data1(seq(split+1:end),1:end);
end




