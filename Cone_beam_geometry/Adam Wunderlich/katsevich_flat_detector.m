clear all; close all; tic
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parameters
Z = -0.25;
% which slice to reconstruct?
usefile = false; % use file for x-ray data?
PIfile = false; % use file for PI-intervals?
show_phantom = false; % plot original phantom?
datfilename = '.\data\test1\Kat_flat\data_zp25f';
PIfilename ='.\data\test1\PIfile_zp25';
mexp = 0;% phantom will be mexp-1 times differentiable
M = 16;% number of detector rows -- take to be even
SourcesPerTurn = 16*M; % number of source positions per turn (even)
L = 4*M;% number of filtering lines on detector plane (even)
ROI = [-1 1 -1 1];
circle = true;% only reconstruct inside circle inscribed in ROI?
MX = 256;% x-dim of reconstruction matrix
MY = 256;% y-dim of reconstruction matrix
height = 0.5; % detector height
delta_w = height/M; % detector element height
delta_u = delta_w; % detector element width
r = 1; % FOV radius
R = 3; % helical scanning radius
D = 6; % source to detector distance
delta = 0;% detector shift (usually either 0 or 1/4)
%P = maxPitch(M,D,R,r,delta_w); % helical pitch
P = .2740;% helical pitch
h = P/(2*pi); % alternate expression for pitch
delta_s = 2*pi/SourcesPerTurn; % stepsize between source positions
alpha_m = asin(r/R); % half fan angle for FOV
LPfiltering = false;% LP filter after Hilbert transform?
w_c = .5*pi;% digital cutoff freq for optional lowpass filtering
pre_interp = false;% use preinterpolation option before backprojection?
Q = 2;% pre-interpolate to a Q-times denser grid
% minimum number of detector columns + 2
minCol = ceil(2*D*tan(alpha_m)/delta_u)+2;
if mod(minCol,2) ~= 0, % make sure minCol is even
    N = minCol+1;
else
    N = minCol;
end
if N < minCol,
    disp('error: number of detector columns too small!')
    return
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% initialization and preprocessing
s_range = [(Z-h*pi)/h,(Z-h*pi)/h+2*pi]; % interval of s values on helix
s = s_range(1):delta_s:(s_range(1)+2*pi); % s samples
s_range(2) = s(SourcesPerTurn+1);
K = length(s); % total number of source positions
hx = (ROI(2)-ROI(1))/(MX-1);
% stepsize for x-dim
hy = (ROI(4)-ROI(3))/(MY-1);
% stepsize for y-dim
x = zeros(MX,1); % range of x-values in reconstruction
y = zeros(1,MY); % range of y-values in reconstruction
x = ROI(1)+hx*[0:MX-1]; y = ROI(3)+hy*[0:MY-1]’;
center = [(ROI(1)+ROI(2)), (ROI(3)+ROI(4))]/2;
% center of ROI
radius = min(ROI(2)-ROI(1),ROI(4)-ROI(3))/2; % radius of inscribed circle
F = zeros(MX,MY); % recontruction


g = zeros(K,N,M); % array of x-ray data -- variables are(s,u,w)
g1 = zeros(K-1,N-1,M); % array of derivatives
%(leave an extra zero in 3rd dimension for interpolation step later)
u = zeros(N,1); % u samples
w = zeros(1,M); % w samples
for j=1:N,
    u(j) = ((j-1)+delta-N/2)*delta_u;
end
for j=1:M,
    w(j) = ((j-1)-M/2)*delta_w;
end
u_grid= zeros(N,M); 
w_grid = zeros(N,M); 
for i=1:M,
    u_grid(:,i) = u;
end
for i=1:N,
    w_grid(i,:) = w;
end
a= zeros(K,3);
% source positions
for i=1:K,
    a(i,1) = R*cos(s(i));
    a(i,2) = R*sin(s(i));
    a(i,3) = h*s(i);
end
% detector coordinate unit vectors
e_u = zeros(3,1); 
e_v = zeros(3,1);
e_w = [0;0;1];
theta = zeros(3,1); % direction unit vector
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute/load cone beam data
if usefile,
    eval(['load ' datfilename]);

else
    disp('computing cone-beam data')
    for m=1:K,
        % s-loop
        if mod(m,50) ==0,
            m
        end
        e_u = [-sin(s(m));cos(s(m));0];
        e_v = [-cos(s(m));-sin(s(m));0];
        for i=1:N, % u-loop
            for j=1:M, % w-loop
                theta = (u(i)*e_u+D*e_v+w(j)*e_w)/sqrt(u(i)^2+D^2+w(j)^2);
                g(m,i,j) = sl3d_xray(a(m,:),theta,mexp);
            end
        end
    end
    eval(['save ' datfilename ' g']);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute derivatives
disp('computing derivatives')
% shift s,u,and w arrays to agree with derivative computation
s = s+delta_s/2;
u = u+delta_u/2; 
w = w+delta_w/2; 
u_grid =u_grid+delta_u/2; 
w_grid = w_grid+delta_w/2;
% use chain rule method to caculate derivative at interlaced positions
% (note that u and w have already been shifted by 1/2 grid cell)
for m=1:(K-1), % s-loop
     for i=1:(N-1), % u-loop
        for j=1:(M-1), % w-loop
        % partial w.r.t. s
        d1 = (g(m+1,i,j) - g(m,i,j) ...
        + g(m+1,i,j+1) - g(m,i,j+1) ...
        + g(m+1,i+1,j) - g(m,i+1,j) ...
        + g(m+1,i+1,j+1) - g(m,i+1,j+1))/(4*delta_s);
        % partial w.r.t. u
        d2 = (g(m,i+1,j) - g(m,i,j) ...            
        + g(m,i+1,j+1) - g(m,i,j+1) ...
        + g(m+1,i+1,j) - g(m+1,i,j) ...
        + g(m+1,i+1,j+1) - g(m+1,i,j+1))/(4*delta_u);
        % partial w.r.t. w
        d3 = (g(m,i,j+1) - g(m,i,j) ...
        + g(m,i+1,j+1) - g(m,i+1,j) ...
        + g(m+1,i,j+1) - g(m+1,i,j) ...
        + g(m+1,i+1,j+1) - g(m+1,i+1,j))/(4*delta_w);
        g1(m,i,j) = d1+(u(i)^2+D^2)/D*d2 + u(i)*w(j)/D*d3;
        end
     end
end
clear g;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% length-correction weighting
g2 = zeros(K-1,N-1,M);
for i=1:(N-1), % u-loop
    for j=1:(M-1), % w-loop
        g2(:,i,j) = g1(:,i,j)*D/sqrt(u(i)^2+D^2+w(j)^2);
    end
end
clear g1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% forward height rebinning
disp('forward height rebinning')
g3 = zeros(K-1,N-1,L);
%variables are (s,u,psi)
delta_psi = (pi+2*alpha_m)/(L-1);
q = M/2; % assume M is even
if mod(M,2) ~= 0,
    disp('error: M not even!');
end
psi = zeros(L,1);
w_k = zeros(N-1,L); 
wf = zeros(L,1); 
k = zeros(L,1); 
t = zeros(L,1);
for j=1:L,
    psi(j) = -pi/2-alpha_m+(j-1)*delta_psi;
end
for i=1:(N-1), % u-loop
    w_k(i,:)=D*h/R*(psi+psi./tan(psi)*u(i)/D);    
end

for m=1:(K-1), % s-loop
    for i=1:(N-1), % u-loop
        % use linear interpolation to find g3(s,u,psi)
        wf = floor(w_k(i,:)/delta_w);
        k = max(1,wf+q+1);
        % set indices less than one to 1
        k = min(M-1,k);
        % set indices greater than M to M
        t = w_k(i,:)/delta_w-wf;
        g3(m,i,:) = squeeze(g2(m,i,k)).*(1-t')+squeeze(g2(m,i,k+1)).*t';
    end
end
clear g2

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1D hilbert transform in u at constant psi
disp('computing 1D hilbert transforms')
g4 = zeros(K-1,N-1,L);
%variables are (s,u,psi)
q = N-1;
h_ideal = zeros(2*q+1,1); %truncated ideal filter response (non-causal)
for n=(-q):q,
    if mod(n,2) == 0, % n even
        h_ideal(n+q+1) = 0;
    else
        % n odd
        h_ideal(n+q+1) = 2/(pi*n);
    end
end
win = real(fftshift(ifft(fftshift(hanning(2*q+1))))); kernel =
conv(win,h_ideal); kernel = kernel(q+1:3*q+1);
for m=1:(K-1), % s-loop
    if mod(m,50)==0,
        m
    end
    for j=1:L, % psi-loop
        g_filt = conv(kernel,g3(m,:,j)); % length 3q
        g4(m,:,j) = g_filt(q+1:2*q);
        % take middle q samples
    end
end
clear g3

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% backward height rebinning
disp('backward height rebinning')
% use linear interpolation as decribed by Noo et al. rather than
% solving the required nonlinear equation
g5 = zeros(K-1,N,M); % variables are (s,u,w)
for m=1:(K-1), % s-loop
    for i=N/2:(N-1), % (positive) u-loop
        for j=1:M,    % w-loop
            for l=1:(L-1), %psi-loop
                if (w(j)>= w_k(i,l) && w(j)<= w_k(i,l+1)),
                    c = (w(j) - w_k(i,l))/(w_k(i,l+1) - w_k(i,l));
                    g5(m,i,j) = (1-c)*g4(m,i,l) + c*g4(m,i,l+1);
                    break
                end
            end % psi-loop
         end % w-loop
    end % u-loop

    for i=1:(N/2-1), % (negative) u-loop
        for j=1:M, % w-loop
            for l=L:-1:2, %psi-loop
                if (w(j)>= w_k(i,l-1) && w(j)<= w_k(i,l)),
                    c = (w(j) - w_k(i,l-1))/(w_k(i,l) - w_k(i,l-1));
                    g5(m,i,j) = (1-c)*g4(m,i,l-1) + c*g4(m,i,l);
                    break;
                end
            end % psi-loop
        end % w-loop
    end % u-loop
end % s-loop
clear g4
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% optional pre-interpolation to a Q-times denser grid
if pre_interp,
    disp('pre-interpolating')
    d_u = delta_u/Q;
    d_w = delta_w/Q;
    ud = (u(1):d_u:u(N))'; % denser grid of u samples
    wd = w(1):d_w:w(M);
    % denser grid of w samples
    u_length = length(ud);
    w_length = length(wd);
    ud_grid = zeros(u_length,w_length);
    wd_grid = zeros(u_length,w_length);
    gi = zeros(K-1,u_length,w_length);

    for j=1:w_length,
        ud_grid(:,j) = ud;
    end
    for j=1:u_length,
        wd_grid(j,:) = wd;
    end
    for j=1:K-1,
        if mod(j,50)==0,
            j
        end
        g5s = zeros(N,M);
        g5s(:,:) = g5(j,:,:);
        gi(j,:,:) = griddata(u_grid,w_grid,g5s(:,:), ...
        ud_grid,wd_grid,’linear’);
    end
else
    u_length = length(u);
    w_length = length(w);
    gi = g5;
    Q=1;
    d_u = delta_u/Q;
    d_w = delta_w/Q;
    ud = u;
    wd = w;
end
%clear g5 u_grid w_grid ud_grid wd_grid
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% find PI-line intervals for each x in reconstruction region
PI = zeros(MX,MY,2); % variables: (x,y,[s_b s_t])
if PIfile,
    eval(['load ' PIfilename ]);
else
    
    disp('finding PI-line intervals with the Champley method')
for i=1:MX,
    for j=1:MY, 
        if circle,
            if (x(i)^2 + y(j)^2) > radius^2,
                continue;
            end
        end
        PI(i,j,:) = findPI(P,R,delta_s,[x(i) y(j) Z]);
    end
end
eval(['save ' PIfilename ' PI']);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% backprojection step
% use trap. rule to implement eqtn(73) in Noo et. al.
disp('backprojecting')
for i=1:MX,    % x-loop
    if mod(i,50)==0,
        i
    end
    for j=1:MY, % y-loop
        if circle,
            if (x(i)^2 + y(j)^2) > radius^2,
                continue;
            end
        end
        sb = PI(i,j,1); % bottom of PI segment
        st = PI(i,j,2); % top of PI segment
        % find indices of s corresponding to sb and st
        k_sb = (sb-s_range(1))/delta_s+1;
        k_st = (st-s_range(1))/delta_s+1;
        % backproject
        for k=(floor(k_sb)-2):(ceil(k_st)+2), % integrate over PI-line
        vstar = R - x(i)*cos(s(k)) - y(j)*sin(s(k));
        ustar = D*(-x(i)*sin(s(k))+y(j)*cos(s(k)))/vstar;
        wstar = D*(Z-h*s(k))/vstar;
        
        % find nearest neightbor
        usn = round(ustar/d_u+(u_length-1)/2+1);
        usn = max(1,usn);
        usn = min(u_length,usn);
        wsn = round(wstar/d_w+(w_length-1)/2+1);
        wsn = max(1,wsn);
        wsn = min(w_length,wsn);
        gi_near = gi(k,usn,wsn);
        % determine rho
        % weight the endpoints of the PI-line in a smooth fashion
        d_in = (s(k)-sb)/delta_s;
        d_out = (st - s(k))/delta_s;
        if (s(k) <= (sb - delta_s))
            rho = 0;
        elseif ((sb - delta_s) < s(k)) && (s(k) <= sb)
            rho = .5*(1+d_in)^2;
        elseif (sb < s(k)) && (s(k) <= (sb+delta_s))
            rho = .5 + d_in - .5*d_in^2;
        elseif (sb + delta_s < s(k)) && (s(k) <= (st - delta_s))
            rho = 1;
        elseif ((st - delta_s) < s(k)) && (s(k) <= st)
            rho = .5 + d_out - .5*d_out^2;
        elseif (st < s(k)) && (s(k) <= (st + delta_s))
            rho = .5*(1+d_out)^2;
        elseif (s(k) > (st + delta_s))
            rho = 0;
        end
        deltaF = rho*delta_s*gi_near/vstar;
        F(i,j) = F(i,j)+deltaF;
        end % end k-loop
    end % end y-loop
end
% end x-loop
F = F./(2*pi);
toc disp(’Done!’)


if show_phantom,
    disp('computing original phantom')
    % plot on MXxMX grid
    dp = 2/(MX-1); % step-size for phantom
    xp = -1:dp:1;
    yp = xp;
    Ph = zeros(MX,MX); % matrix of phantom values
    for k=1:MX,
        for j=1:MX,
            Ph(j,k) = sl3d([xp(k),yp(j) Z],mexp);
        end
    end
    figure
    imagesc(xp,yp,Ph,[0,.07])
    colormap(gray(128));
    axis([-1 1 -1 1])
    axis xy
    axis('square')
    title(['original phantom, z=',num2str(Z)])
    colorbar
end
F = F'; 
figure
imagesc(x,y,F,[0,.07]) 
colormap(gray(128)) 
title(['Katsevich reconstruction, z=',num2str(Z)]) 
colorbar
%load .\data\test1\phantom_zp25
%rel_error = sqrt(sum(sum((Ph-F).^2))/(sum(sum(Ph.^2))));
%rel_error




