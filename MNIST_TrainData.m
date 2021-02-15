clear;

TrainData = matfile('TrainData.mat');
FullData = TrainData.TogetherTrain;

LabelsOne = FullData(:,1);
OutputOne = zeros(10,4000);
for i = 1:4000
    OutputOne(LabelsOne(i)+1,i) = 1;
end

Images = FullData(:,2:785);
Images = Images';
HiddenNeuronOne = 185;
HiddenNeuronTwo = 160;

NoiseCounter = 0;
ImagesLength = length(Images);

for t = 1:1000
    RandNum = randi(4);
    
    switch RandNum
        case 1
            Images(:,t)  = Images(:,t)/10;
        case 2
            Images(:,t)  = Images(:,t)/100;
        case 3
            RandNumTwo = randi(784);
            Images(RandNumTwo,t) = 0;
        case 4
            RandNumTwo = randi(784);
            Images(RandNumTwo,t) = 1;
    end
    NoiseCounter = NoiseCounter + 1;
end

WeightOne = randn(HiddenNeuronOne,784)*sqrt(2/784);
WeightTwo = randn(HiddenNeuronTwo,HiddenNeuronOne)*sqrt(2/HiddenNeuronOne);
WeightThree = randn(10,HiddenNeuronTwo)*sqrt(2/HiddenNeuronTwo);
BiasOne = randn(HiddenNeuronOne,1);
BiasTwo = randn(HiddenNeuronTwo,1);
BiasThree = randn(10,1);

ErrorOne = zeros(10,1);
ErrorTwo = zeros(HiddenNeuronTwo,1);
ErrorThree = zeros(HiddenNeuronOne,1);
ErrorOneCount = zeros(10,1);
ErrorTwoCount = zeros(HiddenNeuronTwo,1);
ErrorThreeCount = zeros(HiddenNeuronOne,1);
GradientOne = zeros(10,1);
GradientTwo = zeros(HiddenNeuronTwo,1);
GradientThree = zeros(HiddenNeuronOne,1);

LearningRate = 0.0058;
EPOCHS = 1100;
MiniBatch = 10;

for k = 1:EPOCHS
    
    if EPOCHS == 256
        tt = 0;
    end
    
    Batches = 1;
    
    for j = 1:4000/MiniBatch
        ErrorOne = zeros(10,1);
        ErrorTwo = zeros(HiddenNeuronTwo,1);
        ErrorThree = zeros(HiddenNeuronOne,1);
        ErrorOneCount = zeros(10,1);
        ErrorTwoCount = zeros(HiddenNeuronTwo,1);
        ErrorThreeCount = zeros(HiddenNeuronOne,1);
        GradientOne = zeros(10,1);
        GradientTwo = zeros(HiddenNeuronTwo,1);
        GradientThree = zeros(HiddenNeuronOne,1);
        for i = Batches:Batches+MiniBatch-1
            
            %Feed forward
            F1 = Images(:,i);
            E2 = WeightOne*F1 + BiasOne;
            F2 = ExopnentialLinearUnit(E2);
            E3 = WeightTwo*F2 + BiasTwo;
            F3 = ExopnentialLinearUnit(E3);
            E4 = WeightThree*F3 + BiasThree;
            F4 = ExopnentialLinearUnit(E4);
            
            %backpropagation
            ErrorOne = (F4-OutputOne(:,i)).*ExopnentialLinearUnitDerivative(E4);
            ErrorTwo = (WeightThree'*ErrorOne).*ExopnentialLinearUnitDerivative(E3);
            ErrorThree = (WeightTwo'*ErrorTwo).*ExopnentialLinearUnitDerivative(E2);
            
            ErrorOneCount = ErrorOneCount + ErrorOne;
            ErrorTwoCount = ErrorTwoCount + ErrorTwo;
            ErrorThreeCount = ErrorThreeCount + ErrorThree;
            GradientOne = GradientOne + ErrorOne*F3';
            GradientTwo = GradientTwo + ErrorTwo*F2';
            GradientThree = GradientThree + ErrorThree*F1';
            
        end
        
        WeightThree = WeightThree - LearningRate/MiniBatch*GradientOne;
        WeightTwo = WeightTwo - LearningRate/MiniBatch*GradientTwo;
        WeightOne = WeightOne - LearningRate/MiniBatch*GradientThree;
        BiasThree = BiasThree - LearningRate/MiniBatch*ErrorOneCount;
        BiasTwo = BiasTwo - LearningRate/MiniBatch*ErrorTwoCount;
        BiasOne = BiasOne - LearningRate/MiniBatch*ErrorThreeCount;
        
        Batches = Batches + MiniBatch;
        
    end
    fprintf('EPOCHs:');
    disp(k)
    [Images,OutputOne] = Shuffle(Images,OutputOne);
end

disp('Training Finished!')
save('WeightThree.mat','WeightThree');
save('WeightTwo.mat','WeightTwo');
save('WeightOne.mat','WeightOne');
save('BiasThree.mat','BiasThree');
save('BiasTwo.mat','BiasTwo');
save('BiasOne.mat','BiasOne');