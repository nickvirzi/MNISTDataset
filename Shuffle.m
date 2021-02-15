function [B,v] = Shuffle(A,y)
cols = size(A,2);
P = randperm(cols);
B = A(:,P);
v = y(:,P);
end