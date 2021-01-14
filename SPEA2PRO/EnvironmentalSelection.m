function [Population,Fitness,ExternalPopulation] = EnvironmentalSelection(Population,ExternalPopulation,N,External_Percentage,GenerationCnt,DeletePercentage,Global)
% The environmental selection of SPEA2

%------------------------------- Copyright --------------------------------
% Copyright (c) 2018-2019 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Calculate the fitness of each solution
    Population = [Population,ExternalPopulation];
    Fitness = CalFitness(Population.objs);

    %% Environmental selection
    Next = Fitness < 1;
    ExternalPopulation = Population(Next(1:(N*External_Percentage)));
    if sum(Next) < N
        [~,Rank] = sort(Fitness);
        Next(Rank(1:N)) = true;
    elseif sum(Next) > N
        Del  = Truncation(Population(Next).objs,sum(Next)-N);
        Temp = find(Next);
        Next(Temp(Del)) = false;
    end
    if mod(GenerationCnt,10) == 0 
        % Delete some weak individuals each 10 iterations
        DeletNumb = floor(N*DeletePercentage);
        NewIndividuals = Global.Initialization(DeletNumb);
        NewIndividualFitness = CalFitness(NewIndividuals.objs);
        [~,Rank] = sort(Fitness);
        Chosed = Next(Rank(1:N-DeletNumb));
        Population = [Population(Chosed),NewIndividuals];
        Fitness    = [Fitness(Chosed),NewIndividualFitness];
    else
        % Population for next generation
        Population = Population(Next);
        Fitness    = Fitness(Next);
    end
end

function Del = Truncation(PopObj,K)
% Select part of the solutions by truncation

    %% Truncation
    Distance = pdist2(PopObj,PopObj);
    Distance(logical(eye(length(Distance)))) = inf;
    Del = false(1,size(PopObj,1));
    while sum(Del) < K
        Remain   = find(~Del);
        Temp     = sort(Distance(Remain,Remain),2);
        [~,Rank] = sortrows(Temp);
        Del(Remain(Rank(1))) = true;
    end
end