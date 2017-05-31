M = 10;
N = 0;
P = 10;
l = [-M/2: M/2-1];
d = 1;
j = [0:P-1];

t = zeros(1,size(l,2));
x = zeros(size(j,2),size(l,2));
y = zeros(size(j,2),size(l,2));
figure,
for i = j+1
    fi = 2*pi*j(i)/P;
    t = d*(l+j(i)*N/P);
    x(i,:) = t*cos(fi);
    y(i,:) = t*sin(fi);
    plot(x(i,:), y(i,:), '.')
    grid on
    hold on
    drawnow
    pause(1)
end
axis equal
title('Standard lattice')
%%
M = 10;
P = 10;
N = P/2;
l = [-M/2: M/2-1];
d = 1;
j = [0:P-1];

t = zeros(1,size(l,2));
x = zeros(size(j,2),size(l,2));
y = zeros(size(j,2),size(l,2));

for i = j+1
    fi = 2*pi*j(i)/P;
    t = d*(l+j(i)*N/P);
    x(i,:) = t*cos(fi);
    y(i,:) = t*sin(fi);
    plot(x(i,:), y(i,:), '.r')
    grid on
    hold on
    drawnow
    pause(1)
end
axis equal
title('Interlaced lattice')



