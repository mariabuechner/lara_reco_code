% phantom.m
% For a given position, compute the value of the 3D phantom
% used for experiments 2 through 7.
% The phantom is m-1 times differentiable
%
% Adam Wunderlich
% last update 8/31/06
function [F] = phantom(x,m)
%Specify parameters of the ellipsoids for 3D phantom.
% a = vector of first half axes
% b = vector of second half axes
% c = vector of third half axes
% x0 = vector of x-coordinates of centers
% y0 = vector of y-coordinates of centers
% z0 = vector of z-coordinates of centers
% alpha = vector of rotation angles (in radians) around the z-axis
% tau = vector of x-ray attentuation coefficients

N = 1; % number of ellipsoids
a= [.35]; b= [.25]; c = [.15]; x0=[.2]; y0=[.3]; z0 = [.1]; 
alpha=[25]*pi/180; tau = [1];

F = 0; y = zeros(1,3); A = zeros(3,3);
for k=1:N,
    % form linear transformation which rotates by -alpha around z-axis
    % and dilates the ellipsoid into a sphere
    A = [1/a(k)*cos(alpha(k)) 1/a(k)*sin(alpha(k)) 0 ; ...
    -1/b(k)*sin(alpha(k)) 1/b(k)*cos(alpha(k)) 0; 0 0 1/c(k)];
    % shift, rotate, and dilate
    y = A*(x' - [x0(k); y0(k); z0(k)]);
    % check to see if new coordinates are in the unit sphere
    t = y(1)^2 + y(2)^2 + y(3)^2;
    if t<= 1,
        F = F + tau(k)*((1-t)^m);
    end
end

