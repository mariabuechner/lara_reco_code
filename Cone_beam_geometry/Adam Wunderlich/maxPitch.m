% maxPitch.m
% find the maximum pitch for a flat detector
% inputs: M = detector rows, D = source to detector distance,
%         R = helix radius, r = FOV radius, d_w = row thickness
% ouput: P = max pitch
% see eqtn. (79) in Noo et al.
%
% Adam Wunderlich
% last update: 10/26/05

function [P] = maxPitch(M,D,R,r,d_w)

alpha_m = asin(r/R); % half fan angle for FOV
u_m = D*tan(alpha_m);

P = (M-1)*(pi*R*D*d_w)/((u_m^2+D^2)*(pi/2+atan(u_m/D)));
