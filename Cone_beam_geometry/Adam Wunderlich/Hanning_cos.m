function [h1] = Hanning_cos(Ns)
h1 = zeros(1,Ns);
for i = [0:Ns-1]
    h1(i+1)= cos(pi*(i-(Ns-1)/2.0-0.5)/(Ns-1))^2;
end
end