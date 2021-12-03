%% Vector fields

% The following models may not be the most efficient. I myself have quiet
% powerful laptop for calculation and great display so I myself do not have
% any problems running this code but some other computer might have


% computer processor: Ryzen 7 5800H
% display: Nvidia RTX3070


%% Task 1
close all
clearvars
clc

%% Solving differential equation with some parameter values and starting points using numerical methods
odef = @(t,y,xc,xp,yc,yp,R0,C0) [y(1).*(1 - y(1)) - xc.*yc.*y(2).*y(1)./(y(1) + R0);
                                 xc.*y(2).*(-1 + yc.*y(1)./(y(1) + R0)) - xp.*yp.*y(2).*y(3)./(y(2) + C0)
                                 xp.*y(3).*(-1 + yp.*y(2)./(y(2) + C0))];
% starting points
R0 = 0.9;
C0 = 0.6;
P0 = 0.2;
% parameter values
xp = 0.7;
xc = 0.6;
yc = 4;
yp = 4;

y0 = [R0,C0,P0];

tspan = [0 1000];

[T,Y] = ode45(@(t,y) odef(t,y,xc,xp,yc,yp,R0,C0), tspan, y0);
figure
hold on
plot(T,Y(:,1),'g','LineWidth',2)
plot(T,Y(:,2),'c','LineWidth',2)
plot(T,Y(:,3),'r','LineWidth',2)
legend('Resorce','Consumer','Predator')


%% 3-D model by using simple vector

% string vector F = F1i + F2j + F3k
% F1 = U = R'
% F2 = V = C'
% F3 = W = P'
U = Y(:,1).*(1 - Y(:,1)) - xc*yc*Y(:,2).*Y(:,1)./(Y(:,1) + R0);
V = xc*Y(:,2).*(-1 + yc*Y(:,1)./(Y(:,1) + R0)) - xp*yp*Y(:,2).*Y(:,3)./(Y(:,2) + C0);
W = xp*Y(:,3).*(-1 + yp*Y(:,2)./(Y(:,2) + C0));

% Y(:,1) = starting points of i component
% Y(:,2) = starting points of j component
% Y(:,3) = starting points of k component
figure
hold on
quiver3(Y(:,1),Y(:,2),Y(:,3),U,V,W,'m','LineWidth',1.2)
plot3(0.7,0.2,0.1286,'r*','LineWidth',3)
xlabel('R')
ylabel('C')
zlabel('P')

%% Vector field for every P0 starting point
figure
hold on
xlabel('R')
ylabel('C')
zlabel('P')
% line to demonstrate that R0 and C0 remains constant
plot3([0.9 0.9],[0.6 0.6],[0 1],'k')

% starting points
R0 = 0.9;
C0 = 0.6;
P0 = [0:0.1:1];
% parameters
xp = 0.7;
xc = 0.6;
yc = 4;
yp = 4;


for i = 1:11

y0 = [R0,C0,P0(i)];

tspan = [0 1000];

[T,Y] = ode45(@(t,y) odef(t,y,xc,xp,yc,yp,R0,C0), tspan, y0);

% Using vectors
% F1 = U = R'
% F2 = V = C'
% F3 = W = P'
U = Y(:,1).*(1 - Y(:,1)) - xc*yc*Y(:,2).*Y(:,1)./(Y(:,1) + R0);
V = xc*Y(:,2).*(-1 + yc*Y(:,1)./(Y(:,1) + R0)) - xp*yp*Y(:,2).*Y(:,3)./(Y(:,2) + C0);
W = xp*Y(:,3).*(-1 + yp*Y(:,2)./(Y(:,2) + C0));

if P0(i) == 0
    marKer = 'g';
    line = 1.5;
elseif P0(i) == 0.2
    marKer = 'm';
    line = 1.5;
else
    marKer = 'b';
    line = 1;
end

% The Plot
% Color green vector for P0 = 0
% Color magenta vector for P0 = 0.2, P0 for research
% Color blue all other P0 starting points
% Red dot, equivalent point for P0 > 0
% Green dot, equivalent point for P0 = 0

quiver3(Y(:,1),Y(:,2),Y(:,3),U,V,W,marKer,'LineWidth',line)
plot3(0.7,0.2,0.1286,'r*','LineWidth',3)
plot3(0.30,0.35,0.0,'g*','LineWidth',3)
xlabel('R')
ylabel('C')
zlabel('P')

end

%% General vector field solution

% following code will plot everything into the same plot and vectorfields
% all equivalent points and from those all asymptotic equivalent points

% Plotting previous vector field
figure
hold on

% starting points
R0 = 0.9;
C0 = 0.6;
P0 = [0:0.1:1];

% parameter values
xp = 0.7;
xc = 0.6;
yc = 4;
yp = 4;

for i = 1:11

y0 = [R0,C0,P0(i)];

tspan = [0 1000];

[T,Y] = ode45(@(t,y) odef(t,y,xc,xp,yc,yp,R0,C0), tspan, y0);

% Using vectors
% F1 = U = R'
% F2 = V = C'
% F3 = W = P'
U = Y(:,1).*(1 - Y(:,1)) - xc*yc*Y(:,2).*Y(:,1)./(Y(:,1) + R0);
V = xc*Y(:,2).*(-1 + yc*Y(:,1)./(Y(:,1) + R0)) - xp*yp*Y(:,2).*Y(:,3)./(Y(:,2) + C0);
W = xp*Y(:,3).*(-1 + yp*Y(:,2)./(Y(:,2) + C0));

if P0(i) == 0
    marKer = 'g';
    line = 1.5;
else
    marKer = 'm';
    line = 1.5;
end

% The Plot
% Color green vector for P0 = 0, R0 = 0.9, C0 = 0.6
% Color magenta vector for P0 > 0, R0 = 0.9, C0 = 0.6 for research
quiver3(Y(:,1),Y(:,2),Y(:,3),U,V,W,marKer,'LineWidth',line)

end


% General vector field for all R0,C0,P0 values with respect of components
% F1, F2 and F3 depending on values R0 and C0 also


n = 5;

% starting points
R0 = linspace(0,1,n);
C0 = linspace(0,1,n);
P0 = linspace(0,1,n);

% parameter values
xp = 0.7;
xc = 0.6;
yc = 4;
yp = 4;

for k = 1:n

for j = 1:n

for i = 1:n

y0 = [R0(k),C0(j),P0(i)];

tspan = [0 1000];

[T,Y] = ode45(@(t,y) odef(t,y,xc,xp,yc,yp,R0(k),C0(j)), tspan, y0);

% Using vectors
% F1 = U = R'
% F2 = V = C'
% F3 = W = P'
U = Y(:,1).*(1 - Y(:,1)) - xc*yc*Y(:,2).*Y(:,1)./(Y(:,1) + R0(k));
V = xc*Y(:,2).*(-1 + yc*Y(:,1)./(Y(:,1) + R0(k))) - xp*yp*Y(:,2).*Y(:,3)./(Y(:,2) + C0(j));
W = xp*Y(:,3).*(-1 + yp*Y(:,2)./(Y(:,2) + C0(j)));

if P0(i) == 0
    marKer = 'g';
    line = 1.5;

else
    marKer = 'b';
    line = 1;
end

% The Plot
% Color green vector for P0 = 0
% Color blue vector for P0 > 0
% % Red dot for research equivalent point equivalent point

quiver3(Y(:,1),Y(:,2),Y(:,3),U,V,W,marKer,'LineWidth',line)
plot3(0.7,0.2,0.1286,'r*','LineWidth',3)
xlabel('R')
ylabel('C')
zlabel('P')

end

end

end

% All equivalent points of vector field exist when gradient is zero,
% Nabla_F = [0]

syms R C P
% F1 = R', F2 = C', F3 = P'

EQ_points = [];
Asymptotic_equivalent_points = [];

% same starting points than the general vector field has
% P0 defined as some value greater than zero since F1, F2 and F3 are not
% dependent of it
R0 = linspace(0,1,n);
C0 = linspace(0,1,n);
P0 = 0.2;

% parameters
xp = 0.7;
xc = 0.6;
yc = 4;
yp = 4;

for m = 1:length(R0)

for n = 1:length(C0)

% Using vectors
% F1 = R'
% F2 = C'
% F3 = P'
F1 = R*(1 - R) - (xc*yc*C*R)/(R + R0(m));
F2 = xc*C*(-1 + (yc*R)/(R + R0(m))) - (xp*yp*C*P)/(C + C0(n));
F3 = xp*P*(-1 + (yp*C)/(C + C0(n)));

% from the task all starting points must not be negative
assume([R>=0,C>=0,P>=0]);

nabla_F = solve(F1 == 0,F2 == 0,F3 == 0,[R,C,P]);

% Definying Jacobian-matrix
J = jacobian([F1 F2 F3],[R,C,P]);

clear F1 F2 F3
F1 = nabla_F.R;
F2 = nabla_F.C;
F3 = nabla_F.P;

% Equivalent points for R0(m), C0(n)
Equivalent_point_matrix = [F1 F2 F3];

E = zeros(size(Equivalent_point_matrix));

% Eigenvalues for testing if equivalent point is asymptotic
for i = 1:size(Equivalent_point_matrix,1)

    E(i,:) = transpose(eig(subs(J,[R C P],Equivalent_point_matrix(i,:))));

end

% interested in from only real parts
E = double(real(E));

Equivalent_point_matrix = double(Equivalent_point_matrix);
EQ_points = [EQ_points;Equivalent_point_matrix];

% if all real parts are less than zero, equivalent point is asymptotic
for i = 1:size(E,1)

   if all(E(i,:) < 0);

     Asymptotic_equivalent_points(end+1,:) = Equivalent_point_matrix(i,:);

   end

end

end

end
% all equivalent points of general vector field including duplicates
EQ_points;
% all aymptotic equivalent points of general vector field, may include
% duplicates
Asymptotic_equivalent_points

% The plot
% Yellow dot, all equivalent points
% Black dot, all asymptotic points, (will plot some of the points on top of yellow points on purpose)

plot3(EQ_points(:,1),EQ_points(:,2),EQ_points(:,3),'y*','LineWidth',2.8)
plot3(Asymptotic_equivalent_points(:,1),Asymptotic_equivalent_points(:,2),Asymptotic_equivalent_points(:,3),'*k','LineWidth',3)



%% Task 2

% Different creator, excluded

%% Task 3
close all
clearvars
clc

%% Test caothic model with given starting points and parameters
odef = @(t,y,xc,xp,yc,yp,R0,C0) [y(1).*(1 - y(1)) - xc.*yc.*y(2).*y(1)./(y(1) + R0);
                                 xc.*y(2).*(-1 + yc.*y(1)./(y(1) + R0)) - xp.*yp.*y(2).*y(3)./(y(2) + C0)
                                 xp.*y(3).*(-1 + yp.*y(2)./(y(2) + C0))];

figure(1)
hold on

figure(2)
hold on

figure(3)
hold on

% given starting points
R0 = 0.161;
C0 = 0.5;
P0 = 0.2;

% given parameters
xp = linspace(0.071,0.225,9);
xc = 0.4;
yc = 2.01;
yp = 5;

% subplotting all to same figures
for i = 1:9

y0 = [R0,C0,P0];

tspan = [0 10000];

[T,Y] = ode45(@(t,y) odef(t,y,xc,xp(i),yc,yp,R0,C0), tspan, y0);

% 2-D plot
figure(1)
subplot(3,3,i)
hold on
plot(T,Y(:,1),'g','LineWidth',.1)
plot(T,Y(:,2),'c','LineWidth',.1)
plot(T,Y(:,3),'r','LineWidth',.1)
legend('Resorce','Consumer','Predator')
title('xp =',num2str(xp(i)))
hold off

t = find(round(T) == 1500);
T2 = T(t(1):t(1)+2000,1);
Y2 = Y(t(1):t(1)+2000,:);

% Zoomed version of plot above
figure(2)
subplot(3,3,i)
hold on
plot(T2,Y2(:,1),'g','LineWidth',.1)
plot(T2,Y2(:,2),'c','LineWidth',.1)
plot(T2,Y2(:,3),'r','LineWidth',.1)
legend('Resorce','Consumer','Predator')
title('xp =',num2str(xp(i)))
hold off

% Using vectors
% F1 = U = R'
% F2 = V = C'
% F3 = W = P'
U = Y(:,1).*(1 - Y(:,1)) - xc*yc*Y(:,2).*Y(:,1)./(Y(:,1) + R0);
V = xc*Y(:,2).*(-1 + yc*Y(:,1)./(Y(:,1) + R0)) - xp(i)*yp*Y(:,2).*Y(:,3)./(Y(:,2) + C0);
W = xp(i)*Y(:,3).*(-1 + yp*Y(:,2)./(Y(:,2) + C0));

% 3-D plot of vector fields
figure(3)
subplot(3,3,i)
hold on
quiver3(Y(:,1),Y(:,2),Y(:,3),U,V,W,'b','LineWidth',1)

xlabel('R')
ylabel('C')
zlabel('P')
title('xp =',num2str(xp(i)))
hold off

end


% Plotting general vector field for the caothic model when xp = 0.071
figure
hold on

n = 5;
% starting points
R0 = linspace(0,1,n);
C0 = linspace(0,1,n);
P0 = linspace(0,1,n);

% parameters
xp = 0.071;
xc = 0.4;
yc = 2.01;
yp = 5;

for k = 1:n

for j = 1:n

for i = 1:n

y0 = [R0(k),C0(j),P0(i)];

tspan = [0 1000];

[T,Y] = ode45(@(t,y) odef(t,y,xc,xp,yc,yp,R0(k),C0(j)), tspan, y0);

% Using vectors
% F1 = U = R'
% F2 = V = C'
% F3 = W = P'
U = Y(:,1).*(1 - Y(:,1)) - xc*yc*Y(:,2).*Y(:,1)./(Y(:,1) + R0(k));
V = xc*Y(:,2).*(-1 + yc*Y(:,1)./(Y(:,1) + R0(k))) - xp*yp*Y(:,2).*Y(:,3)./(Y(:,2) + C0(j));
W = xp*Y(:,3).*(-1 + yp*Y(:,2)./(Y(:,2) + C0(j)));

if P0(i) == 0
    marKer = 'g';
    line = 1.5;

else
    marKer = 'b';
    line = 1;
end

% The Plot
% Color green vector for P0 = 0
% Color blue vector for P0 > 0
quiver3(Y(:,1),Y(:,2),Y(:,3),U,V,W,marKer,'LineWidth',line)

xlabel('R')
ylabel('C')
zlabel('P')

end

end

end




%% Stability of the models defined
%% Task 1
odef = @(t,y,xc,xp,yc,yp,R0,C0) [y(1).*(1 - y(1)) - xc.*yc.*y(2).*y(1)./(y(1) + R0);
                                 xc.*y(2).*(-1 + yc.*y(1)./(y(1) + R0)) - xp.*yp.*y(2).*y(3)./(y(2) + C0)
                                 xp.*y(3).*(-1 + yp.*y(2)./(y(2) + C0))];

% starting points
R0 = 0.9;
C0 = 0.6;
P0 = [0 0.0000000000001 0.2];

% parameters
xp = 0.7;
xc = 0.6;
yc = 4;
yp = 4;

figure
hold on
for i = 1:3
y0 = [R0,C0,P0(i)];

tspan = [0 1000];

[T,Y] = ode45(@(t,y) odef(t,y,xc,xp,yc,yp,R0,C0), tspan, y0);

% Using vectors
% F1 = U = R'
% F2 = V = C'
% F3 = W = P'
U = Y(:,1).*(1 - Y(:,1)) - xc*yc*Y(:,2).*Y(:,1)./(Y(:,1) + R0);
V = xc*Y(:,2).*(-1 + yc*Y(:,1)./(Y(:,1) + R0)) - xp*yp*Y(:,2).*Y(:,3)./(Y(:,2) + C0);
W = xp*Y(:,3).*(-1 + yp*Y(:,2)./(Y(:,2) + C0));

if i == 1
    marKer = 'g';
    line = 1;
elseif i == 2
    marKer = 'k';
else
    marKer = 'm';
    line = 1;

end

% The Plot
% Color green vector for P0 = 0
% Color black vector for P0 = dP
% Color mangenta vector for P0 = 0.2
% Green dot, equivalent point when P0 = 0
% Red dot, equivalent point when P0 > 0

quiver3(Y(:,1),Y(:,2),Y(:,3),U,V,W,marKer,'LineWidth',line)
plot3(0.7,0.2,0.1286,'r*','LineWidth',3)
plot3(0.3,0.35,0,'*g','LineWidth',3)
xlabel('R')
ylabel('C')
zlabel('P')

end



%% Task 3
figure
hold on
n = 5;

% starting ponts
R0 = linspace(0,1,n);
C0 = linspace(0,1,n);
P0 = linspace(0,1,n);

% parameters
xp = 0.0071;
xc = 0.4;
yc = 2.01;
yp = 5;

% General even more caothic vector field
for k = 1:n

for j = 1:n

for i = 1:n

y0 = [R0(k),C0(j),P0(i)];

tspan = [0 1000];

[T,Y] = ode45(@(t,y) odef(t,y,xc,xp,yc,yp,R0(k),C0(j)), tspan, y0);

% Using vectors
% F1 = U = R'
% F2 = V = C'
% F3 = W = P'
U = Y(:,1).*(1 - Y(:,1)) - xc*yc*Y(:,2).*Y(:,1)./(Y(:,1) + R0(k));
V = xc*Y(:,2).*(-1 + yc*Y(:,1)./(Y(:,1) + R0(k))) - xp*yp*Y(:,2).*Y(:,3)./(Y(:,2) + C0(j));
W = xp*Y(:,3).*(-1 + yp*Y(:,2)./(Y(:,2) + C0(j)));

if P0(i) == 0
    marKer = 'g';
    line = 1.5;

else
    marKer = 'b';
    line = 1;
end

% The plot
% Color green vector for P0 = 0
% Color blue vector for P0 > 0

quiver3(Y(:,1),Y(:,2),Y(:,3),U,V,W,marKer,'LineWidth',line)

xlabel('R')
ylabel('C')
zlabel('P')

end

end

end


% 2-D plot from even more caothic model

% starting points
R0 = 0.161;
C0 = 0.5;
P0 = 0.2;

% parameters
xp = 0.0071;
xc = 0.4;
yc = 2.01;
yp = 5;

y0 = [R0,C0,P0];

tspan = [0 10000];

[T,Y] = ode45(@(t,y) odef(t,y,xc,xp,yc,yp,R0,C0), tspan, y0);
figure
hold on
plot(T,Y(:,1),'g','LineWidth',2)
plot(T,Y(:,2),'c','LineWidth',2)
plot(T,Y(:,3),'r','LineWidth',2)
legend('Resorce','Consumer','Predator')
