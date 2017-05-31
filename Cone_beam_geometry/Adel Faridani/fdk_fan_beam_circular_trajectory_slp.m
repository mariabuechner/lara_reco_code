%fdk.m
%Feldkamp-Davis-Kress Algorithm
%for reconstruction on one user defined plane
%specify input parameters here

q=128; %h=1/q = detector spacing
p=pi*q; %number of view angles between 0 and 2*pi
R = 2.868; %Radius of source circle
MX=256; MY= 256; %matrix dimensions

%%%%  Setting the reconstruction plane
% The equation of the plane is <x,nv> = sp
% nv = unit normal vector
% Provide orthonormal unit vecors w1, w2 orthog. to nv.
% Then a point on the plane can be written as
% x = sp*nv + x1*w1 + x2*w2
nv =[0; 0; 1];
sp =-0.25;
w1=[1;0;0];
w2=[0;1;0];

roi=[-1 1 -1 1]; %roi=[xmin xmax ymin ymax]
r = 1.0;
%region of interest where
%reconstruction is computed
circle = 0; % If circle = 1 image computed only inside
% circle inscribed in roi.

dd = 1/q;
dx = 2*r/MX;
fprintf(['Detector spacing: ' num2str(dd) ', Image pixel size: ' num2str(dx)])
% Parameters for mathematical phantom
% centobj - centers of ellipsoids
% axes - length of semiaxis
% rho - densities
% OV - OV(3j-2:3j,1:3) = orthogonal matrix V for j-th object
xe=[0 0 0.22 -0.22 0 0 0 -0.08 0 0.06 0.5538];
ye=[0 -0.0184 0 0 0.35 0.1 -0.1 -0.605 -0.605 -0.605 -0.3858];
ze=[0 0 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25];
ae=[0.69 0.6624 0.11 0.16 0.21 0.046 0.046 0.046 0.023 0.023 0.0333];
be=[0.92 0.874 0.31 0.41 0.25 0.046 0.046 0.023 0.023 0.046 0.206];
ce=[0.9 0.88 0.21 0.22 0.35 0.046 0.02 0.02 0.1 0.1 0.01];
alpha=[0 0 -18 18 0 0 0 0 0 0 -18];
rho= [1 -0.98 -0.02 -0.02 0.01 0.01 0.01 0.01 0.01 0.01 0.03];

%end of input section

ymax = R/sqrt(R^2-r^2);
h = ymax/q;
b=pi/h; rps=1/b;

if MX > 1
    hx = (roi(2)-roi(1))/(MX-1);
    xrange = roi(1) + hx*[0:MX-1];
else
    hx = 0; xrange = roi(1);
end
if MY > 1
    hy = (roi(4)-roi(3))/(MY-1);
    yrange = flipud((roi(3) + hy*[0:MY-1])');
else
    hx = 0; yrange = roi(3);
end

center = [(roi(1)+roi(2)), (roi(3)+roi(4))]/2;
x1 = ones(MY,1)*xrange; %x-coordinate matrix
x2 = yrange*ones(1,MX); %y-coordinate matrix
if circle == 1
    re = min([roi(2)-roi(1),roi(4)-roi(3)])/2;
    chi = ((x1-center(1)).^2 + (x2-center(2)).^2 <= re^2);
    %chi = characteristic function of roi;
else
    chi = isfinite(x1);
end
x1 = x1(chi); x2 = x2(chi);

x3 = sp*nv(3) + w1(3)*x1 + w2(3)*x2;
P = zeros(MY,MX);Pchi = P(chi);
y = h*[-q:q-1];
%ry = 1./sqrt(R.^2 + y.^2);
%s = -R*y.*ry;
bs = b*h*[-2*q:2*q-1];
wb = slkernel(bs)/(rps^2); %compute discrete convolution kernel.
theta = zeros(3,max(size(y)));

for j = 1:p
    j
    Df = zeros(size(y));
    alphaj = (2*pi*(j-1)/p);
    om = [cos(alphaj);sin(alphaj);0];
    a = R*om;
    Q = zeros(2*q+1,2*q+1);
    % Q = zeros(2*q,2*q);
    q1 = 2*q+1;
    for l = 1:2*q %compute line integrals and convolutions
        zl = -ymax + h*(l-1);
        theta(1,:) = -om(2)*y - R*om(1);
        theta(2,:) = om(1)*y - R*om(2);
        theta(3,:) = zl;
        ss = sqrt(sum(theta.^2));

        theta(1,:) = theta(1,:)./ss;
        theta(2,:) = theta(2,:)./ss;
        theta(3,:) = theta(3,:)./ss;
        
        for i = 1:2
            centobj = [xe(i) ye(i) ze(i)];
            axes = [ae(i) be(i) ce(i)];
            density = [rho(i)];
            beta = alpha(i);
            OV =eye(3); 
            OV(2:3,2:3) = [cos(beta) -sin(beta); ...
                           sin(beta) cos(beta)];
            mexp=0;
            RF = Divray(a,theta,centobj,axes,OV,mexp,density);
            RF = RF./sqrt(R^2 + zl^2 + y.^2);
            Df = Df + RF;
        end
        maxdat(l) = max(max(Df));
        % convolution
        C = conv(Df,wb);
        Q(l,1:2*q) = h*C(2*q+1:4*q); 
        %Q(l,(2*q+1))=0;
    end % maxj = max(max(maxdat))
    maxQ = max(max(Q));
    
    % interpolation and backprojection
    u = om'*[w1, w2, nv];
    xom = u(1)*x1 + u(2)*x2 + u(3)*sp;
    up = [-om(2), om(1), 0]*[w1, w2, nv];
    xomp = up(1)*x1 + up(2)*x2 + up(3)*sp;
    %  Q = [real(Q)'; 0];
    rxw = R - xom;
    t = (R*xomp)./rxw;
    zx = R*x3./rxw;
    flz = floor(zx/h);
    l0 = max(1,flz+q+1);
    l0 = min(l0,2*q);
    l01 = min(l0+1,2*q);
    k1 = floor(t/h);
    u = (t/h-k1);
    k = max(1,k1+q+1); 
    k = min(k,2*q);
    tmp1 = ((1-u).*Q(l0+q1*(k-1))+u.*Q(l0+q1*k));
    tmp2 = ((1-u).*Q(l01+q1*(k-1))+u.*Q(l01+q1*k));
    v = zx/h-flz;
    Pupdate = (1-v).*tmp1 + v.*tmp2;
    maxup = max(max(abs(Pupdate)));
    Pchi=Pchi+ Pupdate./(rxw.^2);
end % j-loop
P(chi) = Pchi*((R^3)*2*pi/p);
pmin = min(min(P));
pmax = max(max(P));
window3(pmin,pmax,roi,P); % view the computed image

figure,
plot(P(128,:))