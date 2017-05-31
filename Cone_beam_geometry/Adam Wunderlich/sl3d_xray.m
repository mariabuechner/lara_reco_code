% sl3d_xray.m
% Compute the cone-beam transform of the 3-D Shepp-Logan head phantom.
% inputs: y = vector of source coordinates
%         theta = unit (column) vector pointing in direction of ray
%         m = phantom will be m-1 times differentiable
% output: Df = cone-beam transform
%
% This function is based on Divray by Adel Faridani.
%
% Adam Wunderlich
% last update: 8/20/06

function [Df] = sl3d_xray(y,theta,m)

%Specify parameters of the 12 ellipsoids for 3D S-L phantom.
% a = vector of first half axes
% b = vector of second half axes
% c = vector of third half axes
% x = vector of x-coordinates of centers
% y = vector of y-coordinates of centers
% z = vector of z-coordinates of centers
% alpha = vector of rotation angles (in radians) around the z-axis
% tau = vector of x-ray attentuation coefficients
a= [0.69 0.6624 0.11 0.16 0.21 0.046 0.046 0.046 0.023 0.023];
b= [0.92 0.874 0.31 0.41 0.25 0.046 0.046 0.023 0.023 0.046];
c = [.9 .88 .21 .22 .35 .046 .02 .02 .1 .1];
x0=[0 0 0.22 -0.22 0 0 0 -0.08 0 0.06];
y0=[0 -0.0184 0 0 0.35 0.1 -0.1 -0.605 -0.605 -0.605];
z0 = [0 0 -.25 -.25 -.25 -.25 -.25 -.25 -.25 -.25];
alpha =[0 0 -18 18 0 0 0 0 0 0]*pi/180;
tau = [1 -0.98 -0.02 -0.02 0.01 0.01 0.01 0.01 0.01 0.01];

Df = 0; 
for j = 1:10,
    z = [y(1)-x0(j); y(2)-y0(j); y(3)-z0(j)];
    A = [1/a(j)*cos(alpha(j)) 1/a(j)*sin(alpha(j)) 0 ; ...
        -1/b(j)*sin(alpha(j)) 1/b(j)*cos(alpha(j)) 0; 0 0 1/c(j)];
    Ath = A*theta;
    Az = A*z;
    dot = Az'*Ath;
    naz2 = Az'*Az;
    nath2 = sum(Ath.^2);
    S = naz2 - (dot.^2)./nath2;
    cm = (2^(2*m+1))*(gamma(m+1)^2)/gamma(2*m+2);
    ind = find(S < 1); Sind = S(ind);
    Df(ind) = Df(ind) + tau(j)*cm*((1-Sind).^(m + 0.5))./sqrt(nath2(ind));
end
