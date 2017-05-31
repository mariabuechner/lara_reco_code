function [RF] = Rad(theta,phi,s,x,y,u,v,alpha,rho)
% This function computes the Radon transform of ellipses
% centered at (x,y) with major axis u, minor axis v,
% rotated through angle alpha, with weight rho.
RF = zeros(size(s));
for mu = 1:max(size(x))
    a = (u(mu)*cos(phi-alpha(mu)))^2+(v(mu)*sin(phi-alpha(mu)))^2;
    test = a-(s-[x(mu);y(mu)]'*theta).^2;
    ind = test>0;
    RF(ind) = RF(ind)+rho(mu)*(2*u(mu)*v(mu)*sqrt(test(ind)))/a;
end % mu-loop
