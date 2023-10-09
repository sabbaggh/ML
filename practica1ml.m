clc
clear all
close all
warning off all

inicial = [1; 2; 3; 4; 5];
x = zeros(5,100);
for i = 1:100
    x(:,i) = i*inicial+(randn(5,1));
end

scatter3(x(1,:),x(4,:),x(2,:)'.')

y = x(5,:)';
x = x(1:4,:);
xT = x';
Q = x*xT;
b=-2*x*y;
w0 = randn(4,1);
alfa = 0.001;
ep = 1;

%epsilon = 0.0000001;
while ep > 1e-6
    G = 2*Q*w0+b;
    wn = w0-alfa*G
    ep = sqrt((wn-w0)'*(wn-w0));
    w0 = wn;
end
