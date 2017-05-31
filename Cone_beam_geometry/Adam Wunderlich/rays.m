% rays.m
% Compute the cone-beam transform of the 3-D phantom used
% for experiments 2 through 7.
% inputs: y = vector of source coordinates
%         theta = unit (column) vector pointing in direction of ray
%         m = phantom will be m-1 times differentiable
% output: Df = cone-beam transform
%
% This function is based on Divray by Adel Faridani.
%
% Adam Wunderlich
% last update: 8/31/06
function [Df] = rays(y,theta,m)
% Specify parameters of the ellipsoids for phantom.
% a = vector of first half axes
% b = vector of second half axes
% c = vector of third half axes
% x = vector of x-coordinates of centers
% y = vector of y-coordinates of centers
% z = vector of z-coordinates of centers
% alpha = vector of rotation angles (in radians) around the z-axis
% tau = vector of x-ray attentuation coefficients
N = 1; % number of ellipsoids
a= [.35]; b= [.25]; c = [.15]; x0=[.2]; y0=[.3]; z0 = [.1]; 
alpha=[25]*pi/180; tau = [1];
Df = 0; 

for j = 1:N,
    z = [y(1)-x0(j); y(2)-y0(j); y(3)-z0(j)];
    A = [1/a(j)*cos(alpha(j)) 1/a(j)*sin(alpha(j)) 0 ; ...
         -1/b(j)*sin(alpha(j)) 1/b(j)*cos(alpha(j)) 0; 0 0 1/c(j)];
    Ath = A*theta;
    Az = A*z;
    dot = Az'*Ath; % <Az,Ath>
    naz2 = Az'*Az; % ||Az||^2
    nath2 = sum(Ath.^2); % ||Ath||^2
    S = naz2 - (dot.^2)./nath2;
    cm = (2^(2*m+1))*(gamma(m+1)^2)/gamma(2*m+2);
    ind = find(S < 1); Sind = S(ind);
    Df(ind) = Df(ind) + tau(j)*cm*((1-Sind).^(m + 0.5))./sqrt(nath2(ind));
end
