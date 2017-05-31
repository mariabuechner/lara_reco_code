% slkernel.m
% compute scaled Shepp-Logan Kernel
% written by Adel Faridani

function [k] = slkernel(s,r)
s = s/r;
k = zeros(size(s));

% if denominator is small, return 1/pi
small = abs((pi/2)^2-s.^2)<=1.e-6; % indices of small denominators
k(small) = ones(size(k(small)))/pi;
not_small = abs((pi/2)^2-s.^2)>1.e-6; % indices of not small denoms
t = s(not_small);
k(not_small)=1/(r^2)*1/(2*pi^3)*(pi/2 - t.*sin(t))./((pi/2)^2 - t.^2);
