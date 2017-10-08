data = csvread('data.csv');

Alpha = .03;

[train,test] = distributeData(data,0.75);
[weight,error1] = hardTraining(train,Alpha);
error2 = testing(test,weight);

figure
plot (data(1:2000,1),data(1:2000,2),'+b',data(2001:4000,1),data(2001:4000,2),'+r')
hold on;
x = 50 : 1 : 85 ;
y = -1*(weight(1,3)+weight(1,1)*x)/weight(1,2);
plot(x,y)
title1 = strcat('Hard Perceptron with 75% training: Error=',num2str(error2(1,1)));
title(title1)
xlabel('Height (in)')
ylabel('Weight (lbs)')

[train,test] = distributeData(data,0.50);
[weight,error1] = hardTraining(train,Alpha);
error2 = testing(test,weight);

figure
plot (data(1:2000,1),data(1:2000,2),'+b',data(2001:4000,1),data(2001:4000,2),'+r')
hold on;
x = 50 : 1 : 85 ;
y = -1*(weight(1,3)+weight(1,1)*x)/weight(1,2);
plot(x,y)
title1 = strcat('Hard Perceptron with 50% training: Error=',num2str(error2(1,1)));
title(title1)
xlabel('Height (in)')
ylabel('Weight (lbs)')

[train,test] = distributeData(data,0.25);
[weight,error1] = hardTraining(train,Alpha);
error2 = testing(test,weight);

figure
plot (data(1:2000,1),data(1:2000,2),'+b',data(2001:4000,1),data(2001:4000,2),'+r')
hold on;
x = 50 : 1 : 85 ;
y = -1*(weight(1,3)+weight(1,1)*x)/weight(1,2);
plot(x,y)
title1 = strcat('Hard Perceptron with 25% training: Error=',num2str(error2(1,1)));
title(title1)
xlabel('Height (in)')
ylabel('Weight (lbs)')

[train,test] = distributeData(data,0.75);
[weight,error1] = softTraining(train,Alpha);
error2 = testing(test,weight);

figure
plot (data(1:2000,1),data(1:2000,2),'+b',data(2001:4000,1),data(2001:4000,2),'+r')
hold on;
x = 50 : 1 : 85 ;
y = -1*(weight(1,3)+weight(1,1)*x)/weight(1,2);
plot(x,y)
title1 = strcat('Soft Perceptron with 75% training: Error=',num2str(error2(1,1)));
title(title1)
xlabel('Height (in)')
ylabel('Weight (lbs)')

[train,test] = distributeData(data,0.50);
[weight,error1] = softTraining(train,Alpha);
error2 = testing(test,weight);

figure
plot (data(1:2000,1),data(1:2000,2),'+b',data(2001:4000,1),data(2001:4000,2),'+r')
hold on;
x = 50 : 1 : 85 ;
y = -1*(weight(1,3)+weight(1,1)*x)/weight(1,2);
plot(x,y)
title1 = strcat('Soft Perceptron with 50% training: Error=',num2str(error2(1,1)));
title(title1)
xlabel('Height (in)')
ylabel('Weight (lbs)')

[train,test] = distributeData(data,0.25);
[weight,error1] = softTraining(train,Alpha);
error2 = testing(test,weight);

figure
plot (data(1:2000,1),data(1:2000,2),'+b',data(2001:4000,1),data(2001:4000,2),'+r')
hold on;
x = 50 : 1 : 85 ;
y = -1*(weight(1,3)+weight(1,1)*x)/weight(1,2);
plot(x,y)
title1 = strcat('Soft Perceptron with 25% training: Error=',num2str(error2(1,1)));
title(title1)
xlabel('Height (in)')
ylabel('Weight (lbs)')

function [training,testing_neuron] = distributeData(data1,proportion)
%This function returns two randomly distributed data sets as training and 
%testing based on proportion (that should be training).
    n = size(data1,1);
    split = round(n*proportion);
    seq = randperm(n);
    training = data1(seq(1:split),1:end);
    testing_neuron = data1(seq(split+1:end),1:end);
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
        desired = zeros(np,1);
        for j = 1:np
            output = trainingData(j,1)*weights(1,1)+trainingData(j,2)*weights(1,2)+weights(1,3);
            if output > 0
                output = 1;
            else
                output = -1;
            end
            if trainingData(j,3) == 1
                desired(j,1) = 1;
            else
                desired(j,1) = -1;
            end
            delta = alpha*(desired(j,1)-output);
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
        error = sum((desired(:,1)-outputTotal(:,1)).^2);
        i = i+1;
    end
end

function [weights,error] = softTraining(trainingData,learningConst)
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
        desired = zeros(np,1);
        for j = 1:np
            
            X = trainingData(j,1)*weights(1,1)+trainingData(j,2)*weights(1,2)+weights(1,3);
            
            output = tanh(X);
            
            if trainingData(j,3) == 1
                desired(j,1) = 1;
            else
                desired(j,1) = -1;
            end
            delta = alpha*(desired(j,1)-output);
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
        error = sum((desired(:,1)-outputTotal(:,1)).^2);
        i = i+1;
    end
end

function error = testing(testingData,perceptron)
    np = size(testingData,1);
    weights = perceptron;
    desired = zeros(np,1);
    for i = 1:np
        if testingData(i,3) == 1
            desired(i,1) = 1;
        else
            desired(i,1) = -1;
        end
    end
    outputTotal = testingData(:,1)*weights(1,1)+testingData(:,2)*weights(1,2)+weights(1,3);
    for i = 1:np
       if outputTotal(i,1) > 0
           outputTotal(i,1) = 1;
       else
           outputTotal(i,1) = -1;
       end
    end
    error = sum((desired(:,1)-outputTotal(:,1)).^2);
end