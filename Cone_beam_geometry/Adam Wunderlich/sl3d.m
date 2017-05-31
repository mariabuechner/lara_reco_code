% sl3d.m
% For a given position, compute the value of the 3D Shepp-Logan phantom.
% The phantom is m-1 times differentiable
%
% Adam Wunderlich
% last update 5/18/06
function [F] = sl3d(x,m)

%Specify parameters of the 12 ellipsoids for 3D S-L phantom.
% a = vector of first half axes
% b = vector of second half axes
% c = vector of third half axes
% x0 = vector of x-coordinates of centers
% y0 = vector of y-coordinates of centers
% z0 = vector of z-coordinates of centers
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

F = 0; y = zeros(1,3); A = zeros(3,3);
for k=1:10,
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
