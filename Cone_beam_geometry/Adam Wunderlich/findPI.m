% findPI.m
%
% Find the the parametric interval corresponding to the unique PI-line
% passing through the point x for a given helical pitch.
% This code implements the method of Kyle Champley.
% inputs: P = pitch (cm/turn), R = helix radius, delta_s = s stepsize, x
% output: PI = [sb st]
%
% Adam Wunderlich
% last update: 5/18/06

function [PI] = findPI(P,R,delta_s,x)
h = P/(2*pi);
r = sqrt(x(1)^2+x(2)^2); 
gamma = atan2(x(2),x(1));

options = optimset('TolX',h*delta_s/100,'FunValCheck','on');
[sb,fval,exitflag] = fzero(@(sb) PIfun(r,R,h,gamma,x(3),sb),[(x(3)-h*pi)/h,x(3)/h],options);
if (exitflag ~=1)
    disp('Error: PI invalid');  
end

% note that beta=sb in Kyle’s formula
alphaX = atan(r*sin(gamma-sb)/(R - r*cos(gamma-sb)));
st = sb + pi - 2*alphaX;
PI = [sb st];
