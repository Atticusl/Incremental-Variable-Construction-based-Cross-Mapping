%% multi-species discrete Lotka-Volterra competition model
function [G, delay, standard] = multi_dis_LV_model(N, maxL, noise, rand_delay, structure, num)
% Data generation of multi-species discrete Lotka-Volterra competition model
%   Input
%           N           number of species
%           maxL        length of series
%           noise
%           rand_delay  choice of random interaction delay, 'True' or 'False'
%           structure   choice of structural modes, which is 1, 2, or 3.
%           num         number of random interactions
%   Output
%           G           time series with length L * species n
%           delay       time delay matrix
%           standard	ground truth of interactions

G=[];
k = 0; maxk = 100;
while isempty(G)
    [initial_value, interaction, delay] = get_para(N, rand_delay, structure, num);
    [G, standard] = benchmark_delay(maxL, initial_value, interaction, delay, noise);
    k = k + 1;
    if k == maxk
       fprintf('Large generation!\n')
    end
end
end

function [initial_value, interaction, delay] = get_para(N, rand_delay, structure, num)
initial_value = 0.2+0.8*rand(N,1);
if structure == 1
    % cycle mode
    interaction = zeros(N);
    interaction(1:N+1:end)=3.4+0.4*rand(N,1);
    interaction([N,N+1:N+1:end])= 0.1+0.1*rand(N,1);
else
    interaction=rand(N); interaction(1:N+1:end)=0;
    if structure == 2
        % random mode
        for i = 1:N
            [~,I] = sort(interaction(i,:),'descend');
            interaction(i,I(1:num)) = 0.1+0.1*rand(num,1); interaction(i,I(num+1:end)) = 0;
        end
        interaction(1:N+1:end)=3.4+0.4*rand(N,1);
    elseif structure == 3
        % sturcture stochastic mode
        interaction([N,N+1:N+1:end])= 0;
        for i = 1:N
            [~,I] = sort(interaction(i,:),'descend');
            interaction(i,I(1:num)) = 0.8+0.1*rand(num,1); interaction(i,I(num+1:end)) = 0;
        end
        interaction(1:N+1:end)=3.4+0.4*rand(N,1);
        interaction([N,N+1:N+1:end])= 0.1+0.1*rand(N,1);
    end
end

delay = diag(ones(N,1));
[row,col] = find((interaction-diag(diag(interaction)))~=0);
switch rand_delay
    case 'True'
        delay(sub2ind(size(delay), row, col)) = round(rand(1,length(row))*5) + 1; % random interaction delay
    case 'False'
        delay(sub2ind(size(delay), row, col)) = 1;
end
end

function [data, standard] = benchmark_delay(maxL, initial_value, interaction, delay, noise)
%   Input
%           maxL            length of series
%           initial_value   initial value for each species
%           interaction     interaction matrix
%           delay           time delay matrix
%           noise           
%   Output
%           G               time series with length L * species n
%           standard        ground truth of interactions
N = length(initial_value); m = 50;
G = zeros(maxL+m, N); G(1,:) = initial_value;
standard = interaction - diag(diag(interaction));
% generate prefix series
for t = 2:m
    G(t,:) = diag(G(t-1,:)) * diag(diag(interaction)) * (1 - G(t-1,:))' + normrnd(0,noise,N,1);
end
% generate series
for t = m+1:maxL+m
    for i = 1:N
        E = 0;
        for j = 1:N
            E = interaction(j,i) * G(t-delay(j,i),j) + E;
        end
        G(t,i) = G(t-delay(i,i),i)*(interaction(i,i) - E) + normrnd(0,noise);
    end
    if any(isnan(G(t,:))) || ~isempty(find(G(t,:) <= 0,1))
        G=[];
        break
    end
end
if ~isempty(G)
    data = G(m+1:maxL+m,:); % Delete prefix series

else
    data = [];
end
end