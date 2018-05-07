fileID = fopen('lenses.txt','r');
formatSpec = '%d %f';
sizeA = [5 Inf];
A = fscanf(fileID,formatSpec,sizeA);
fclose(fileID);

fileID = fopen('weight.txt','r');
formatSpec = '%f';
sizeB = [1 Inf];
w = fscanf(fileID,formatSpec,sizeB);
w = w';
fclose(fileID);

fileID = fopen('bias.txt','r');
formatSpec = '%f';
sizeC = [1 Inf];
b = fscanf(fileID,formatSpec,sizeC);
fclose(fileID);

dataSet = A';
trainSet = dataSet(1:18,1:4)/5; 
trainTarget = dataSet(1:18,end)/5;
testSet = dataSet(19:24,1:4)/5;
testTarget = dataSet(19:24,end)/5;

j = 1;

while(j<=size(testSet,1))
        neth1 = testSet(j,1) * w(1,:) + testSet(j,2) * w(4,:) + testSet(j,3) * w(7,:)' + testSet(j,4) * w(10,:) + b(1);
        fNeth1 = 1 / (1 + exp(-neth1));
        
        neth2 = testSet(j,1) * w(2,:) + testSet(j,2) * w(5,:)' + testSet(j,3) * w(8,:) + testSet(j,4) * w(11,:) + b(2);
        fNeth2 = 1 / (1 + exp(-neth2));
        
        neth3 = testSet(j,1) * w(3,:) + testSet(j,2) * w(6,:) + testSet(j,3) * w(9,:) + testSet(j,4) * w(12,:) + b(3);
        fNeth3 = 1 / (1 + exp(-neth3));
        
        neto1 = fNeth1 * w(13,:) + fNeth2 * w(14,:) + fNeth3 * w(15,:) + b(4);
        fNeto1 = 1 / (1 + exp(-neto1));
        
        error = testTarget(j,:) - fNeto1;
        fprintf('Hedef : %.8f -- Çýkýþ : %.8f -- Hata : %.8f\n',testTarget(j,:)*5,fNeto1*5,error*5);
    j = j + 1;
end     
