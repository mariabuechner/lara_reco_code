function Df = Divray(a,theta,center,axes,OV,mexp,density)
% Integrals over lines through point z in directions theta
% Cf. problem 9 in Appendix C.
% a = source position
% theta: columns of theta are directions of rays
% center: rows of center are transposed of center points
% axes: rows of axes contain length of principlal axes
% OV: OV(3j-2:3j,1:3) is orthonornmal matrix for j-th object
% mexp: vector with exponents (mexp = 0 for ellipsoids)

N = max(size(mexp)); % Number of objects in phantom
Df = zeros(size(theta(1,:)));
for j = 1:N
    z = a - center(j,:)';
    A = diag(1./axes(3*j-2:3*j))*OV(3*j-2:3*j,:);
    Ath = A*theta;
    Az = A*z;
    dot = Az'*Ath;
    naz2 = Az'*Az;
    nath2 = sum(Ath.^2);
    S = naz2 - (dot.^2)./nath2;
    m = mexp(j);
    cm = (2^(2*m+1))*(gamma(m+1)^2)/gamma(2*m+2);
    ind = find(S < 1); Sind = S(ind);
    tmp = density(j)*cm*((1-Sind).^(m + 0.5))./sqrt(nath2(ind));
    Df(ind) = Df(ind) + tmp;
end
