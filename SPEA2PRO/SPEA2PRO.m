function SPEA2PRO(Global)
% <algorithm> <S>
% Strength Pareto evolutionary algorithm 2

%------------------------------- Reference --------------------------------
% E. Zitzler, M. Laumanns, and L. Thiele, SPEA2: Improving the strength
% Pareto evolutionary algorithm, Proceedings of the Fifth Conference on
% Evolutionary Methods for Design, Optimization and Control with
% Applications to Industrial Problems, 2001, 95-100.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2018-2019 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Generate random population
    Population = Global.Initialization();
    Fitness    = CalFitness(Population.objs);
    ExternalPopulation = Global.Initialization(1); % Add a external population 
    GenerationCnt = 1; % Count the iteration number of generation
    %% Optimization
    while Global.NotTermination(Population)
        MatingPool = TournamentSelection(2,Global.N,Fitness);
        Offspring  = GA(Population(MatingPool));
        [Population,Fitness,ExternalPopulation] = EnvironmentalSelection([Population,Offspring],ExternalPopulation,Global.N,0.25,GenerationCnt,0.10,Global);
        GenerationCnt = GenerationCnt +1;
    end
end