data = csvread('data.csv');
[train,test] = distributeData(data,0.75);
[weight,error1] = hardTraining(train,0.03);

function [training,testing] = distributeData(data1,proportion)
%This function returns two randomly distributed data sets as training and 
%testing based on proportion (that should be training).
    n = size(data1,1);
    split = round(n*proportion);
    seq = randperm(n);
    training = data1(seq(1:split),1:end);
    testing = data1(seq(split+1:end),1:end);
end

function [weights,error] = hardTraining(trainingData,learningConst)
%This function trains a perceptron on the training data (<=1000 
%iterations) and returns the weights of the resulting perceptron
    np = size(trainingData,1);
    ite = 1000;
    alpha = learningConst;
    epsilon = 0.00001;
    i = 1;
    error = 1; %default value
    weights = sum(rand(3)); %randomized starting weights
    while (i <= ite) && (error > epsilon)
        desiredTotal = zeros(np,1);
        for j = 1:np
            output = trainingData(j,1)*weights(1,1)+trainingData(j,2)*weights(1,2)+weights(1,3);
            if output > 0
                output = 1;
            else
                output = -1;
            end
            if trainingData(j,3) == 1
                desiredTotal(j,1) = 1;
            else
                desiredTotal(j,1) = -1;
            end
            delta = alpha*(desiredTotal(j,1)-output);
            deltaW = trainingData(j,:);
            deltaW(1,3) = 1;
            deltaW = deltaW*delta;
            weights = weights+deltaW;
        end
        outputTotal = trainingData(:,1)*weights(1,1)+trainingData(:,2)*weights(1,2)+weights(1,3);
        for j = 1:size(outputTotal,1)
           if outputTotal(j,1) > 0
               outputTotal(j,1) = 1;
           else
               outputTotal(j,1) = -1;
           end
        end
        error = sum((desiredTotal(:,1)-outputTotal(:,1)).^2);
        i = i+1;
    end
end