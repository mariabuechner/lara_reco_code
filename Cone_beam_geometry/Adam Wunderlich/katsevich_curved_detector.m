% curved_slice.m
% Cone-beam image reconstruction with sources on a helix
% of fixed radius with a curved detector. The FBP algorithm is based
% on Katsevich’s exact inversion formula, using the implementation steps
% described by Noo et. al. in
% "Exact Helical Reconstruction using Native Cone-beam Geometries,"
% Physics in Medicine and Biology, Vol. 48, (2003) p.3787-3818.
% This script computes a single slice of the phantom for a given z value.
% Note: The phantom lies inside the unit sphere.
% Adam Wunderlich
% last update: 9/6/06

%clear all
close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parameters
Z = -0.25; % which slice to reconstruct?
usefile = false; % use file for x-ray data?
usecanopy = false;
PIfile = false; % use file for PI-intervals
show_phantom = true; % plot original phantom?
% datfilename = 'projection_data';
% PIfilename = 'PIfile';
datfilename = 'data_zp25c';
PIfilename = 'PIfile_zp25c';
mexp = 0; % phantom will be mexp-1 times differentiable
M = 16; % number of detector rows -- take to be even
N = 138; % number of detector columns -- take to be even
%SourcesPerTurn = 16*M; % number of source positions per turn (even)
L = 4*M; % number of filtering curves on detector plane (even)
ROI = [-1 1 -1 1];
circle = true; % only reconstruct inside circle inscribed in ROI?
r = 1; % FOV radius
R = 3; % helical scanning radius
D = 6; % source to detector distance
delta = 0; % detector shift (usually either 0 or 1/4)
height = 0.5; % detector height
delta_w = height/M; % detector element height
d_alpha = delta_w; % detector element width
delta_alpha = d_alpha/D;
Pmax = maxPitch2(M,D,R,r,delta_w); % helical pitch
P = Pmax; %.2740; % helical pitch
h = P/(2*pi); % alternate expression for pitch
delta_s = d_alpha;%2*pi/SourcesPerTurn; % stepsize between source positions
MX = 256; % x-dim of reconstruction matrix
MY = 256; % y-dim of reconstruction matrix
alpha_m = asin(r/R); % half fan angle for FOV
w_c = .5*pi; % digital cutoff freq for optional lowpass filtering
LPfiltering= false; % LP filter after Hilbert transform?
pre_interp = false; % use preinterpolation option before backprojection?
Q = 2; % pre-interpolate to a Q-times denser grid
minCol = ceil(2*alpha_m/delta_alpha)+2; % minimum number of detector columns
%if mod(minCol,2) ~= 0, % make sure minCol is even
%    N = minCol+1;
% else 
%    N = minCol;
%end
if N < minCol,
    disp('error: number of detector columns too small!')
    return
end

if usecanopy,
    eval(['load ' datfilename]);  
    K = 302; % number of source points
    L = 64; % number of k-lines
    MX = 256; % x-dim of reconstruction matrix
    MY = 256; % y-dim of reconstruction matrix
    M = 16;
    N = 138;
    r = 1; % FOV radius
    R = 3; % helical scanning radius
    D = 6; % source to detector distance
    for m=1:K,    % s-loop
        if mod(m,50) ==0,
            m
        end
        temp = squeeze(Df(m,:,:));
        g(m,:,:) = fliplr(temp');
    end
    eval(['save ' datfilename '_g g']);
end

 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% initialization and preprocessing
s_range = [(Z-h*pi)/h,Z/h+2*pi]; % interval of s values on helix
s = s_range(1):delta_s:(s_range(2)); % s samples
s_range(2) = s(end);
K = length(s); % total number of source positions
hx = (ROI(2)-ROI(1))/(MX-1); % stepsize for x-dim
hy = (ROI(4)-ROI(3))/(MY-1); % stepsize for y-dim
x = zeros(MX,1); % range of x-values in reconstruction
y = zeros(1,MY); % range of y-values in reconstruction
x = ROI(1)+hx*[0:MX-1];
y = ROI(3)+hy*[0:MY-1]';
center = [(ROI(1)+ROI(2)), (ROI(3)+ROI(4))]/2; % center of ROI
radius = min(ROI(2)-ROI(1),ROI(4)-ROI(3))/2; % radius of inscribed circle
F = zeros(MX,MY); % recontruction
%g = zeros(K,N,M); % array of x-ray data -- variables are(s,alpha,w)
g1 = zeros(K-1,N-1,M); % array of derivatives
%(leave an extra zero in 3rd dimension for interpolation step later)

% find PI-line intervals for each x in reconstruction region
PI = zeros(MX,MY,2); % variables: (x,y,[s_b s_t])
if PIfile,
    eval(['load ' PIfilename ]);
else
    disp('finding PI-line intervals with the Champley method')
    tic
    for i=1:MX,
        for j=1:MY,
            if circle,
                % only reconstruct over inscribed circle?
                if (x(i)^2 + y(j)^2) > radius^2,
                    continue;
                end
            end
            PI(i,j,:) = findPI(P,R,delta_s,[x(i) y(j) Z]);
        end
    end
    toc
    eval(['save ' PIfilename ' PI']);
end


alpha = zeros(N,1); % alpha samples
w = zeros(1,M); % w samples
for j=1:N,
    alpha(j) = ((j-1)+delta-N/2)*delta_alpha;
end
for j=1:M,
    w(j) = ((j-1)-(M-1)/2)*delta_w;
end

alpha_grid= zeros(N,M); 
w_grid = zeros(N,M); 
for i=1:M,
    alpha_grid(:,i) = alpha;
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
e_u = zeros(3,1); e_v = zeros(3,1); e_w = [0;0;1];
theta = zeros(3,1); % direction unit vector
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute/load cone beam data
if usecanopy,
    disp('Uses my forward projection!')
elseif usefile,
    eval(['load ' datfilename]);
else
    disp('computing cone-beam data')
    tic
    for m=1:K,    % s-loop
        if mod(m,50) ==0,
            m
        end
        e_u = [-sin(s(m));cos(s(m));0];
        e_v = [-cos(s(m));-sin(s(m));0];
        for i=1:N, % alpha-loop
            for j=1:M, % w-loop
                theta = (D*sin(alpha(i))*e_u+D*cos(alpha(i))*e_v+w(j)*e_w)/sqrt(D^2+w(j)^2);
                g(m,i,j) = sl3d_xray(a(m,:),theta,mexp);
            end
        end
    end
    toc
    eval(['save ' datfilename ' g']);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute derivatives
tic
disp('computing derivatives')
% shift s, alpha, and w arrays to agree with derivative computation
s = s+delta_s/2;
alpha = alpha+delta_alpha/2; 
alpha_grid = alpha_grid+delta_alpha/2;
% use chain rule method
for m=1:(K-1), % s-loop
    for i=1:(N-1), % alpha-loop
        for j=1:M, % w-loop
            % partial w.r.t. s
            d1 = (g(m+1,i,j) - g(m,i,j) ...
            + g(m+1,i+1,j) - g(m,i+1,j))/(2*delta_s);
            % partial w.r.t. alpha
            d2 = (g(m,i+1,j) - g(m,i,j) ...
            + g(m+1,i+1,j) - g(m+1,i,j))/(2*delta_alpha);
            g1(m,i,j) = d1+d2;
        end
    end
end
toc
eval(['save ' datfilename  '_der g1']);

clear g;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% length-correction weighting
tic
disp('Cos (lambda) scaling')
g2 = zeros(K-1,N-1,M);
for j=1:(M), % w-loop
    g2(:,:,j) = g1(:,:,j)*D/sqrt(D^2+w(j)^2);
end
toc
eval(['save ' datfilename  '_g2 g2']);
clear g1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% forward height rebinning
tic
disp('forward height rebinning')
g3 = zeros(K-1,N-1,L);
%variables are (s,alpha,psi)
delta_psi = (pi+2*alpha_m)/(L-1);
q = double(M/2); % assume M is even
if mod(M,2) ~= 0,
    disp('error: M not even!');
end

psi = zeros(L,1); 
w_k = zeros(N-1,L);
wf = zeros(L,1); 
k =  zeros(L,1); 
t = zeros(L,1); 
for j=1:L,
    psi(j) = -pi/2-alpha_m+(j-1)*delta_psi;
end
for i=1:(N-1), % alpha-loop
    w_k(i,:)=D*h/R*(psi*cos(alpha(i))+psi./tan(psi)*sin(alpha(i)));
end


for m=1:(K-1), % s-loop
    for i=1:(N-1), % alpha-loop
        % use linear interpolation to find g3(s,alpha,psi)
        wf = floor(w_k(i,:)/delta_w);
        k = max(1,wf+q);
        % set indices less than one to 1
        k = min(M-1,k);
        % set indices greater than M to M
        t = (w_k(i,:)-w(k))/delta_w;
        g3(m,i,:) = squeeze(g2(m,i,k)).*(1-t')+squeeze(g2(m,i,k+1)).*t';
    end
end
toc
eval(['save ' datfilename  '_fhr g3']);
clear g2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1D hilbert transform in u at constant psi
% use etqns (51) and (53) in Noo et. al.
tic
disp('computing 1D hilbert transforms')
g4 = zeros(K-1,N-1,L);
%variables are (s,alpha,psi)
q = N-1;
h_ideal = zeros(2*q+1,1); %truncated ideal filter response (non-causal)
for n=(-q):q,
    if mod(n,2) == 0, % n even
        h_ideal(n+q+1) = 0;
    else % n odd
        h_ideal(n+q+1) = 2.0/(pi*n);
    end
end
win = real(fftshift(ifft(ifftshift(hanning(2*q+1))))); 
kernel = conv(h_ideal,win); 
kernel = kernel(q+1:3*q+1);
for m=1:(K-1), % s-loop
    for j=1:L, % psi-loop
        g_filt = conv(kernel,g3(m,:,j)); % length 3q
        g4(m,:,j) = g_filt(q+1:2*q);     % take middle q samples
    end
end
eval(['save ' datfilename  '_ht g4']);
clear g3


% % 1D hilbert transform in u at constant psi
% % use etqns (51) and (53) in Noo et. al.
% disp('computing 1D hilbert transforms')
% g4 = zeros(K-1,N-1,L);
% %variables are (s,alpha,psi)
% q = N-1;
% h_ideal = zeros(2*q+1,1); %truncated ideal filter response (non-causal)
% for n=(-q):q,
%     if mod(n-0.5,2) == 0, % n even
%         h_ideal(n+q+1) = 0;
%     else % n odd
%         h_ideal(n+q+1) = (1-cos(pi*(n-0.5)))/(pi*(n-0.5));
%     end
% end
% win = real(fftshift(ifft(ifftshift(Hanning_cos(2*q+1))))); 
% kernel = conv(h_ideal,win); 
% kernel = kernel(q+1:3*q+1);
% for m=1:(K-1), % s-loop
%     for j=1:L, % psi-loop
%         g_filt = conv(kernel,g3(m,:,j)); % length 3q
%         g4(m,:,j) = g_filt(q+1:2*q);     % take middle q samples
%     end
% end
% eval(['save ' datfilename  '_ht g4']);
% clear g3
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% optional lowpass filtering
if LPfiltering,
    disp('lowpass filtering in alpha at constant psi')
    h_ideal = zeros(2*q+1,1); %truncated ideal LP filter response (non-causal)
    for n=-q:q,
        if n~=0,
            h_ideal(n+q+1) = sin(w_c*n)./(pi*n);
        else
            h_ideal(n+q+1) = 1;
        end
    end
    win = hanning(2*q+1); % window
    h_lp = win.*h_ideal; % LP filter impulse response
    for m=1:(K-1), % s-loop
        for j=1:L, % psi-loop
            g_filt = conv(h_lp,g4(m,:,j)); % length 3q
            g4(m,:,j) = g_filt(q+1:2*q);  % take middle q samples
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% backward height rebinning
tic
disp('backward height rebinning')
% use linear interpolation as decribed by Noo et al. rather than
% solving the required nonlinear equation
g5 = zeros(K-1,N,M); % variables are (s,alpha,w)

for m=1:(K-1), % s-loop
    for i=N/2:(N-1), % (positive) alpha-loop
        for j=1:M,    % w-loop
            for l=1:(L-1), % psi-loop
                if (w(j)>= w_k(i,l) && w(j)<= w_k(i,l+1)),
                    c = (w(j) - w_k(i,l))/(w_k(i,l+1) - w_k(i,l));
                    g5(m,i,j) = (1-c)*g4(m,i,l) + c*g4(m,i,l+1);
                    break
                end
            end % psi-loop
        end % w-loop
    end % u-loop
    for i=1:(N/2-1),  % (negative) alpha-loop
        for j=1:M,    % w-loop
            for l=L:-1:2, % psi-loop
                if (w(j)>= w_k(i,l-1) && w(j)<= w_k(i,l)),
                    c = (w(j) - w_k(i,l-1))/(w_k(i,l) - w_k(i,l-1));
                    g5(m,i,j) = (1-c)*g4(m,i,l-1) + c*g4(m,i,l);
                    break;
                end
            end % psi-loop
        end % w-loop
    end % u-loop
end % s-loop
toc
eval(['save ' datfilename  '_bhr g5']);
clear g4
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% post-cosine weighting
tic
disp('post-cosine weighting')
for m=1:(K-1), %s-loop
    for j=1:M, % w-loop
        g5(m,:,j) = cos(alpha)'.*g5(m,:,j);
    end
end
toc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% optional pre-interpolation to a Q-times denser grid
if pre_interp,
    disp('pre-interpolating')
    
    delta_alpha2 = delta_alpha/Q;
    delta_w2 = delta_w/Q;
    alphad = (alpha(1):delta_alpha2:alpha(N))';    % denser grid of alpha samples
    wd = w(1):delta_w2:w(M);                       % denser grid of w samples
    alpha_length = length(alphad);
    w_length = length(wd);
    alphad_grid = zeros(alpha_length,w_length);
    wd_grid = zeros(alpha_length,w_length);
    gi = zeros(K-1,alpha_length,w_length);
    
    for j=1:w_length,
        alphad_grid(:,j) = alphad;
    end
    for j=1:alpha_length,
        wd_grid(j,:) = wd;
    end
    for j=1:K-1,
        if mod(j,50)==0,
            j
        end
        g5s = zeros(N,M);
        g5s(:,:) = g5(j,:,:);
        gi(j,:,:) = griddata(alpha_grid,w_grid,g5s(:,:), ...
        alphad_grid,wd_grid,'linear');
    end
else
    alpha_length = length(alpha);
    w_length = length(w);
    gi = g5;
    Q=1;
    delta_alpha2 = delta_alpha/Q;
    delta_w2 = delta_w/Q;
    alphad = alpha;
    wd = w;
end
clear g5 alpha_grid w_grid alphad_grid wd_grid


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% backprojection step
% use trap. rule to implement eqtn(73) in Noo et. al.
tic
disp('backprojecting') 
for i=1:MX,
    if mod(i,50)==0,
        i
    end
    for j=1:MY,
        if circle,
            % only reconstruct over inscribed circle?
            if (x(i)^2 + y(j)^2) > radius^2,
                continue;
            end
        end
        
        sb = PI(i,j,1);
        % bottom of PI segment
        st = PI(i,j,2);
        % top of PI segment
        % find indices of s corresponding to sb and st
        k_sb = (sb-s_range(1))/delta_s+1;
        k_st = (st-s_range(1))/delta_s+1;
        % backproject
        for k=(floor(k_sb)-2):(ceil(k_st)+2), % integrate over PI-line
            vstar = R - x(i)*cos(s(k)) - y(j)*sin(s(k));
            alpha_star = atan((-x(i)*sin(s(k))+y(j)*cos(s(k)))/vstar);
            wstar = D*cos(alpha_star)*(Z-h*s(k))/vstar;    % find nearest neightbor
            alpha_sn = round(alpha_star/delta_alpha2+(alpha_length-1)/2+1);
            alpha_sn = max(1,alpha_sn);
            alpha_sn = min(alpha_length,alpha_sn);
            wsn = round(wstar/delta_w2+(w_length-1)/2+1);
            wsn = max(1,wsn);
            wsn = min(w_length,wsn);
            gi_near = gi(k,alpha_sn,wsn);

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
            deltaF = 1*delta_s*gi_near/vstar;
            F(i,j) = F(i,j)+deltaF;
        end % end PI-interval loop
    end % end y-loop
end% end x-loop
F = F./(2*pi);
toc
disp('Done!')
if show_phantom,
    disp('computing original phantom')
    N = double(MX); % plot on MXxMX grid
    dp = 2/(N-1); % step-size for phantom
    xp = -1:dp:1;
    yp = xp;
    Ph = zeros(N,N); % matrix of phantom values
    for k=1:N,
        for j=1:N,
            Ph(j,k) = sl3d([xp(k),yp(j) Z], 0);
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
figure,
imagesc(x,y,F,[0,0.07]) % axis xy axis image
colormap(gray(128)) 
title(['Katsevich reconstruction, z=', num2str(Z)])
colorbar

%load phantom_zp25
%rel_error = sqrt(sum(sum((Ph-F).^2))/(sum(sum(Ph.^2))));
%rel_error

