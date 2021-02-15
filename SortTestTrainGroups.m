clear;

Images = load('MNISTnumImages5000_balanced.txt');
Labels = load('MNISTnumLabels5000_balanced.txt');

Together = [Labels Images];
TogetherCopy = Together;
counter1 = 1;
counter2 = 0;

c0 = 0; c1 = 0; c2 = 0; c3 = 0; c4 = 0; c5 = 0; c6 = 0; c7 = 0; c8 = 0; c9 = 0;

LoopCounter = false;

TogetherTrain = zeros(4000, 785);

for i = 5000:-1:1001
    while LoopCounter == false
        RandNum = randi(i);
        NUMB = Together(RandNum,1);
        
        switch NUMB
            case 0
                if c0 < 400
                    c0 = c0 + 1;
                    LoopCounter = true;
                else
                    LoopCounter = false;
                end
            case 1
                if c1 < 400
                    c1 = c1 + 1;
                    LoopCounter = true;
                else
                    LoopCounter = false;
                end
            case 2
                if c2 < 400
                    c2 = c2 + 1;
                    LoopCounter = true;
                else
                    LoopCounter = false;
                end
            case 3
                if c3 < 400
                    c3 = c3 + 1;
                    LoopCounter = true;
                else
                    LoopCounter = false;
                end
            case 4
                if c4 < 400
                    c4 = c4 + 1;
                    LoopCounter = true;
                else
                    LoopCounter = false;
                end
            case 5
                if c5 < 400
                    c5 = c5 + 1;
                    LoopCounter = true;
                else
                    LoopCounter = false;
                end
            case 6
                if c6 < 400
                    c6 = c6 + 1;
                    LoopCounter = true;
                else
                    LoopCounter = false;
                end
            case 7
                if c7 < 400
                    c7 = c7 + 1;
                    LoopCounter = true;
                else
                    LoopCounter = false;
                end
            case 8
                if c8 < 400
                    c8 = c8 + 1;
                    LoopCounter = true;
                else
                    LoopCounter = false;
                end
            case 9
                if c9 < 400
                    c9 = c9 + 1;
                    LoopCounter = true;
                else
                    LoopCounter = false;
                end
        end
    end
    
    if counter1 == 4000
        LoopCounter = true;
    else
        LoopCounter = false;
    end
    
    TogetherTrain(counter1,:) = Together(RandNum,:);
    counter1 = counter1 + 1;
    Together(RandNum,:) = [];
end

save('TrainData.mat', 'TogetherTrain');
save('TestData.mat', 'Together');

