% PIfun.m
% Evaluate a function used to find the PI-line, using Kyle Champley’s
% method.
% 
% Adam Wunderlich
% last update: 5/18/06

function y = PIfun(r,R,h,gamma,x3,sb)
temp = R - r*cos(gamma-sb); 
y = h*((pi - 2*atan(r*sin(gamma-sb)/temp))*(1 + (r^2 - R^2)/(2*R*temp))+ sb) - x3;
