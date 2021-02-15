
TestData = matfile('TestData.mat');
TestData = TestData.Together;

LabelsTwo = TestData(:,1);
OutputTwo = zeros(10,1000);
for i = 1:1000
    OutputTwo(LabelsTwo(i)+1,i) = 1;
end

images = TestData(:,2:785);
images2 = images;

images = images';

WeightThree = matfile('WeightThree.mat');
WeightFour = WeightThree.WeightThree;
WeightTwo = matfile('WeightTwo.mat');
WeightFive = WeightTwo.WeightTwo;
WeightOne = matfile('WeightOne.mat');
WeightSix = WeightOne.WeightOne;
BiasThree = matfile('BiasThree.mat');
BiasFour = BiasThree.BiasThree;
BiasTwo = matfile('BiasTwo.mat');
BiasFive = BiasTwo.BiasTwo;
BiasOne = matfile('BiasOne.mat');
BiasSix = BiasOne.BiasOne;
Success = 0;
NumberOfElem = 1000;
Matrix = zeros(10,10);

for i = 1:NumberOfElem
    OutputArrayOne = ExopnentialLinearUnit(WeightSix*images(:,i)+BiasSix);
    OutputArrayTwo = ExopnentialLinearUnit(WeightFive*OutputArrayOne+BiasFive);
    OutputArrayThree = ExopnentialLinearUnit(WeightFour*OutputArrayTwo+BiasFour);
    
    CorrectNumber = find(OutputArrayThree == max(OutputArrayThree));
    CorrectNumber = CorrectNumber - 1;
    
    if LabelsTwo(i) == CorrectNumber
        Success = Success + 1;
    end
    
    %Creates confusion matrix
    Index = LabelsTwo(i);
    Cheeks = Matrix(CorrectNumber + 1, Index + 1);
    Cheeks = Cheeks + 1;
    Matrix(CorrectNumber + 1, Index + 1) = Cheeks;
    
    
end

figure;

for K = 1:20
    subplot(4,5,K);
    temp = images2(K * 50,:);
    temp2 = reshape(temp,28,28);
    imshow(temp2);
end


fprintf('Accuracy: ');
fprintf('%f',Success/NumberOfElem*100);
disp(' %');