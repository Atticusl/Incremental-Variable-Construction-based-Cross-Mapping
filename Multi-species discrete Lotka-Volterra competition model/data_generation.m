%%
clear
maxL = 100; noise = 0.002; rand_delay = 'True'; trails = 100;
for N = [10,20,30,40,50,60,70,80,90,100]
    cyc_data=[]; cyc_delay=[]; cyc_st=[];
    for i = 1:trails
        [cyc_data(:,:,i), cyc_delay(:,:,i), cyc_st(:,:,i)] = multi_dis_LV_model(N, maxL, noise, rand_delay, 1, 1);
        fprintf('cyc_data: %2.0f \n',i);
    end
%     name = sprintf('cyc_N_%d_trails_%d.mat',N,trails);
%     save(name, 'cyc_data', 'cyc_delay', 'cyc_st')
end

%%
clear
maxL = 100; noise = 0.002; rand_delay = 'True'; trails = 50;
for N = [10,20,30,40,50,60,70,80,90,100]
    ran_data=[]; ran_delay=[]; ran_st=[];
    for i = 1:trails
        [ran_data(:,:,i), ran_delay(:,:,i), ran_st(:,:,i)] = multi_dis_LV_model(N, maxL, noise, rand_delay, 2, 1);
        fprintf('ran_data: %2.0f \n',i);
    end
%     name = sprintf('ran_N_%d_trails_50.mat',N);
%     save(name, 'ran_data', 'ran_delay', 'ran_st')
end

%%
clear
maxL = 100; noise = 0.002; rand_delay = 'True'; trails = 100;
for N = [10,20,30,40,50,60,70,80,90,100]
    ssm_data=[]; ssm_delay=[]; ssm_st=[];
    for i = 1:trails
        [ssm_data(:,:,i), ssm_delay(:,:,i), ssm_st(:,:,i)] = multi_dis_LV_model(N, maxL, noise, rand_delay, 3, 1);
        fprintf('ssm_data: %2.0f \n',i);
    end
%     name = sprintf('ssm_N_%d_trails_%d.mat',N,trails);
%     save(name, 'ssm_data', 'ssm_delay', 'ssm_st')
end