%{
divray.m function: The idea behind this function is that we 
compute intersection length of a ray with an ellipsoid by utilizing a 
linear transform (operator matrix A) which transforms ellipsoid into a unit
sphere. Source point and ray direction theta get also transformed.
We compute intersection of this transformed ray with unit sphere. 
Once we have intersection length through unit sphere, we can scale the 
length and get the intersection length through ellipsoid.
 
f value for a particular ray is computed as:
f => density * cm  * (1-Sind)^(m+0.5) /norm2(nath)
If I accept m = 0 for unit ball, then
f => density * cm * sqrt(1-Sind) /norm2(nath)
where density corresponds to attenuation coefficient,  cm is always 2 for m = 0.
I interpret cm as intersection of a line passing through unit sphere along 
the Az vector, this line always passes through sphere center. If I scale 
cm * sqrt(1-Sind), I get intersection length of ray with angle Ath and unit
sphere. And finally scaling it with norm of Ath gives me intersection 
length of ray at angle theta and ellipsoid.

Here is a 2D example
%}

clear all
close all

% create ellipsis
fi = [0:0.005:2*pi];
a = 0.5; % ellipse parameter
b = 0.25;% ellipse parameter
c = [0.5 0.5]; % ellipse center
angle = pi/3; % ellipse rotation angle
xs = cos(2*pi*fi);
ys = sin(2*pi*fi);

A = [cos(angle) -sin(angle); sin(angle) cos(angle)]*[a 0; 0 b];
ell = zeros(1,2);
xe = nan(size(xs));
ye = nan(size(ys));
for i = 1: length(fi)
    ell = A * [xs(i) ys(i)]'; % transform circle coordinates into ellipse
    xe(i) = ell(1)+c(1);      % shift the center, move  (0,0) -> (cx, cy) 
    ye(i) = ell(2)+c(2);      
end
hold on
plot(xe,ye)
plot(c(1), c(2), '.r')
plot(0,0,'.k')
grid on

% source and x-ray
y = [-0.2 0.2];
plot(y(1), y(2), '.b')
theta = [0.02 0.01]/sqrt(0.0004+0.0001); % theta is unit vector
t = 0:0.05:3;
x = repmat(y, length(t), 1) + repmat(t',1,2).*repmat(theta,length(t),1);
x = x'; % x-ray
plot(x(1,:), x(2,:), 'g')



% inverse process (ellipsis-> unit sphere)
invA = [1/a 0; 0 1/b]*[cos(angle) sin(angle); -sin(angle) cos(angle)]; % same as inv(A)
es = zeros(1,2);
xes = nan(size(xs));
yes = nan(size(ys));
for i = 1: length(fi)
    es = invA*[xe(i) ye(i)]'; % transform shifted ellipse coordinates into unit sphere coordinates
    xes(i) = es(1); 
    yes(i) = es(2);
end
hold on
plot(xes,yes)
axis([-3 3 -3 3])
axis equal
grid on


ci = (invA*c')'; %(center of ellipsis-> center of unit sphere)
ti = (invA*theta')'; %(ray direction -> A transformed direction)
yi = (invA*y')'; %(souce-> A transfromed source)
xi = repmat(yi, length(t), 1) + repmat(t',1,2).*repmat(ti,length(t),1);
xi = xi'; %(A transformed ray)
plot(ci(1), ci(2), '.r')
plot(yi(1), yi(2), '.b')
plot(xi(1,:), xi(2,:), 'g')

% compute intersection line with unit sphere
% distance d -distance from ci to inverted ray d = (A(yi-ci)xti)/norm(ti)
% temp0, temp1 -first two points of x-ray
temp0 = xi(:,1); 
temp1 = xi(:,2);
d = sqrt(sum(cross(([ci 0]'-[temp0;0]),([temp0; 0]-[temp1; 0])).^2))/sqrt(sum((temp0-temp1).^2));
li = 2*sqrt(1-d^2); % intersection with unit sphere

% compute intesection line on ellipse
%    le           li
% --------- =  ----------, ||theta|| = 1
%  ||theta||    ||ti||
scaling = sqrt(sum(theta.^2))/sqrt(sum(ti.^2));
le = li * scaling; % intersection with ellipsis