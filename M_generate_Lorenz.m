
addpath('\klw\Research\Functions');

dim = 3;
Lorenz_sigma = 10;
Lorenz_beta = 8/3;


t_cut = 600;
t_max = 180 + t_cut;
t_step_plot = 0.03;
t_step_ratio = 15;

% Lya ~ 1

%{
rho_mean = 27;
rho_std = 2;
rho_min = 24.5;
rho_max = 29.5;
%}

%rho_mean = 30;
%rho_std = 5;
rho_min = 25;
rho_max = 35;


para_dist_type = 2;
% 1: croped Gaussian
% 2: uniformed

repeat_num = 500;

u_set = zeros(repeat_num, dim, (t_max-t_cut)/t_step_plot );
para_real_set = zeros(repeat_num,1);
rng('shuffle');
tic
parfor repeat_i = 1:repeat_num
    if para_dist_type == 1
        rho = 0;
        while rho<rho_min || rho>rho_max
            rho = rho_std*randn+rho_mean;
        end
    elseif para_dist_type == 2
        rho = rho_min + rand*(rho_max-rho_min);
    end
    
    x0 = [ 28 * rand - 14; 30 * rand - 15; 20 * rand];    
    flag = [Lorenz_sigma rho Lorenz_beta];
    [~,x] = ode4(@(t,x) eq_Lorenz(t,x,flag),0:(t_step_plot/t_step_ratio):t_max,x0);
    x = x(1:t_step_ratio:end,:);
    x = x(round( t_cut / t_step_plot + 2):end,:);
    u_set(repeat_i,:,:) = x';
    para_real_set(repeat_i) = rho;
    fprintf('%d\n',repeat_i)
end
toc

d_mean_set = zeros(size(u_set,2),1);
d_std_set = zeros(size(u_set,2),1);
for d_i = 1:size(u_set,2)
    u_d = reshape(u_set(:,d_i,:),[1,size(u_set,1)*size(u_set,3)]);
    d_mean = mean(u_d);
    d_std = std(u_d);
    u_set(:,d_i,:) = (u_set(:,d_i,:) - d_mean) / d_std;
    
    d_mean_set(d_i) = d_mean;
    d_std_set(d_i) = d_std;
end
clear u_d rho

%% plot
label_font_size = 12;
ticks_font_size = 12;

figure()
histogram(para_real_set,20)
xlabel('rho')
ylabel('number')
set(gcf,'color','white')

%% plot a trial
plot_trial = randi(repeat_num);
plot_dim = 3;
x = zeros(dim,size(u_set,3));
x(:,:) = u_set(plot_trial,:,:);
rho = para_real_set(plot_trial);

figure()
plot3(x(1,:),x(2,:),x(3,:))
xlabel("x")
ylabel("y")
zlabel("z")
set(gcf,'color','white')
title(['rho = ' num2str(rho)])

figure()
plot(t_step_plot*(1:size(u_set,3)),x(3,:));
xlabel('t')
ylabel('x')
title(['rho =' num2str(rho,8)])
xlabel('t','FontSize',label_font_size)
ylabel('x','FontSize',label_font_size)
set(gca,'FontSize',ticks_font_size)
set(gcf,'color','white')

figure()
plot(x(3,:))
set(gcf,'color','white')

clear x 
%% save
%save('save_Lorenz_25_35_500_step6000_test.mat')

%save('save_Lorenz_245_345_1000_step6000_train.mat')
%save('save_Lorenz_245_345_500_step6000_test.mat')