% maxPitch2.m
% find the maximum pitch for a curved detector
% inputs: M = detector rows, D = source to detector distance,
%         R = helix radius, r = FOV radius, d_w = row thickness
% ouput: P = max pitch
% see eqtn. (37) in Noo et al.
%
% Adam Wunderlich
% last update: 11/7/05

function [P] = maxPitch2(M,D,R,r,d_w)

alpha_m = asin(r/R); % half fan angle for FOV
u_m = D*tan(alpha_m);

P = (M-1)*pi*R*d_w*cos(alpha_m)/(D*(pi/2+alpha_m));
