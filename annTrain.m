fileID = fopen('lenses.txt','r');
formatSpec = '%d %f';
sizeA = [5 Inf];
A = fscanf(fileID,formatSpec,sizeA);
dataSet = A';
trainSet = dataSet(1:18,1:4)/5; % 5 is used to normalize.Because we use sigmoid function that is range between 0 and 1
trainTarget = dataSet(1:18,end)/5;
testSet = dataSet(19:24,1:4)/5;
testTarget = dataSet(19:24,end)/5;

min = -1;
max = 0.1;
w = (max-min).*rand(15,1) + min;
b= (max-min).*rand(1,4) + min;

errorRateo1 = 0;
errorRateh1 = 0;
errorRateh2 = 0;
errorRateh3 = 0;

learningRate = 0.7;
momentumCoefficient = 0.7;

changeWeightBias = [0,0,0,0];
changeWeight = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0];

j = 1;
i = 0;
count = 0;
while(j<=size(trainSet,1))
        neth1 = trainSet(j,1) * w(1,:) + trainSet(j,2) * w(4,:) + trainSet(j,3) * w(7,:)' + trainSet(j,4) * w(10,:) + b(1);
        fNeth1 = 1 / (1 + exp(-neth1));
        
        neth2 = trainSet(j,1) * w(2,:) + trainSet(j,2) * w(5,:)' + trainSet(j,3) * w(8,:) + trainSet(j,4) * w(11,:) + b(2);
        fNeth2 = 1 / (1 + exp(-neth2));
        
        neth3 = trainSet(j,1) * w(3,:) + trainSet(j,2) * w(6,:) + trainSet(j,3) * w(9,:) + trainSet(j,4) * w(12,:) + b(3);
        fNeth3 = 1 / (1 + exp(-neth3));
        
        neto1 = fNeth1 * w(13,:) + fNeth2 * w(14,:) + fNeth3 * w(15,:) + b(4);
        fNeto1 = 1 / (1 + exp(-neto1));
        
        error = trainTarget(j,:) - fNeto1;
        fprintf('Target : %.8f -- Output : %.8f -- Error : %.8f -- Success Rate : %.2f',trainTarget(j,:)*5,fNeto1*5,error*5,100-(abs(error*5*100)));
        
        if(abs(error) > 0.005)
            i = 0;
            errorRateo1 = fNeto1 * (1-fNeto1) * error;         

            changeWeight(13) = learningRate * errorRateo1 * fNeth1 + momentumCoefficient * changeWeight(13);
            changeWeight(14) = learningRate * errorRateo1 * fNeth2 + momentumCoefficient * changeWeight(14);
            changeWeight(15) = learningRate * errorRateo1 * fNeth3 + momentumCoefficient * changeWeight(15);
            changeWeightBias(4) = learningRate * errorRateo1 * 1 + momentumCoefficient * changeWeightBias(4);

            errorRateh1 = fNeth1 * (1-fNeth1) * errorRateo1 * w(13,:);
            errorRateh2 = fNeth2 * (1-fNeth2) * errorRateo1 * w(14,:);
            errorRateh3 = fNeth3 * (1-fNeth3) * errorRateo1 * w(15,:);

            changeWeight(1) = learningRate * errorRateh1 * trainSet(j,1) + momentumCoefficient * changeWeight(1);
            changeWeight(2) = learningRate * errorRateh2 * trainSet(j,1) + momentumCoefficient * changeWeight(2);
            changeWeight(3) = learningRate * errorRateh3 * trainSet(j,1) + momentumCoefficient * changeWeight(3);
            changeWeight(4) = learningRate * errorRateh1 * trainSet(j,2) + momentumCoefficient * changeWeight(4);
            changeWeight(5) = learningRate * errorRateh2 * trainSet(j,2) + momentumCoefficient * changeWeight(5);
            changeWeight(6) = learningRate * errorRateh3 * trainSet(j,2) + momentumCoefficient * changeWeight(6);
            changeWeight(7) = learningRate * errorRateh1 * trainSet(j,3) + momentumCoefficient * changeWeight(7);
            changeWeight(8) = learningRate * errorRateh2 * trainSet(j,3) + momentumCoefficient * changeWeight(8);
            changeWeight(9) = learningRate * errorRateh3 * trainSet(j,3) + momentumCoefficient * changeWeight(9);
            changeWeight(10) = learningRate * errorRateh1 * trainSet(j,4) + momentumCoefficient * changeWeight(10);
            changeWeight(11) = learningRate * errorRateh2 * trainSet(j,4) + momentumCoefficient * changeWeight(11);
            changeWeight(12) = learningRate * errorRateh3 * trainSet(j,4) + momentumCoefficient * changeWeight(12);
            changeWeightBias(1) = learningRate * errorRateh1 * 1 + momentumCoefficient * changeWeightBias(1);
            changeWeightBias(2) = learningRate * errorRateh2 * 1 + momentumCoefficient * changeWeightBias(2);
            changeWeightBias(3) = learningRate * errorRateh3 * 1 + momentumCoefficient * changeWeightBias(3);

            w(13,:) = w(13,:) + changeWeight(13);
            w(14,:) = w(14,:) + changeWeight(14);
            w(15,:) = w(15,:) + changeWeight(15);
            b(4) = b(4) + changeWeightBias(4);

            w(1,:) = w(1,:) + changeWeight(1);
            w(4,:) = w(4,:) + changeWeight(4);
            w(7,:) = w(7,:) + changeWeight(7);
            w(10,:) = w(10,:) + changeWeight(10);

            w(2,:) = w(2,:) + changeWeight(2);
            w(5,:) = w(5,:) + changeWeight(5);
            w(8,:) = w(8,:) + changeWeight(8);
            w(11,:) = w(11,:) + changeWeight(11);

            w(3,:) = w(3,:) + changeWeight(3);
            w(6,:) = w(6,:) + changeWeight(6);
            w(9,:) = w(9,:) + changeWeight(9);
            w(12,:) = w(12,:) + changeWeight(12);
            
            b(1) = b(1) + changeWeightBias(1);
            b(2) = b(2) + changeWeightBias(2);
            b(3) = b(3) + changeWeightBias(3);
             
        
        else
            i = i + 1;
            if(i == 18)
                disp('Network Learned !');
                break;
            end
        end

        
        disp(j);
        j=j+1;
        
        if(j==19)
            j=1;
        end
    count = count + 1;
    md = mod(count,500000);
    if(md == 0)
        if (learningRate > 0.5)
            learningRate = learningRate - 0.1;
        end
    end
    if(count == 4000000)
        disp('Maximum iteration count is reached');
        break;
    end
end

w=w';

fileID = fopen('weight.txt','w');
fprintf(fileID,'%.8f\n',w);
fclose(fileID);
fileID = fopen('bias.txt','w');
fprintf(fileID,'%.8f\n',b);
fclose(fileID);

