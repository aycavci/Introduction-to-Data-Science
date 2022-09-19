% Genetic algorithm for the selection of the best subset of features
% Course: Introduction to Data Science
% Author: George Azzopardi
% Date:   October 2019

function bestchromosome = myGeneticAlgorithm(features,labels)
% features = A matrix of independent variables
% labels = A vector that contains the labels for each rows in matrix features


nchroms       = 100; % number of chromosomes
nepochs       = 10;  % number of epochs
nparentsratio = 0.2; % portion of elite list
mutateprob    = 0.1; % probability to mutate a bit in a chromosome

% Create figure that shows the progress of the genetic algorithm
% figure;hold on;
% title('Feature Selection with Genetic Algorithm');
% colorlist = jet(nepochs);

% Convert labels, which can be in string format, to numeric.
[lbls,h] = grp2idx(labels);

% Iterate through all epochs
for epoch = 1:nepochs
%     fprintf('epoch %d of %d\n',epoch,nepochs);
    if epoch == 1
        % generate the intial popultion of chromosome with randomly
        % assigned bits
        pop = generateInitialPopulation(nchroms,size(features,2));        
    else
        % generate a new population by creating offspring from the best
        % performing chromosome (or parents)
        pop = getnewpopulation(pop,score,nparentsratio,mutateprob);
    end    
    pop = logical(pop);
    
    % Compute the fitness score for each chromosome
    score = zeros(1,nchroms);
    for i = 1:nchroms
        score(i) = getScore(pop(i,:),features,lbls);    
    end    
    
    % Plot the scores to visualize the progress
    plot(sort(score,'descend'),'color',colorlist(epoch,:));
    xlabel('Chromosome');
    ylabel('Fitness Score');
    legendList{epoch} = sprintf('Epoch %d',epoch);
    legend(legendList);
    drawnow;
end

% Return the chromosome with the maximum fitness score
[~,mxind] = max(score);
bestchromosome = pop(mxind,:);
end

function newpop = getnewpopulation(pop,score,nparentsratio,mutateprob)
% Generate a new population by first selecting the best performing
% chromosomes from the given pop matix, and subsequently generate new offspring chromosomes from randomly
% selected pairs of parent chromosomes.

% Step 1. Write code to select the top performing chromosomes. Use nparentsratio to
% calculate how many parents you need. If pop has 100 rows and
% nparentsration is 0.2, then you have to select the top performing 20
% chromosomes
[~,ind] = sort(score,'descend');
nparents = nparentsratio * size(pop,1);

newpop = zeros(size(pop));
newpop(1:nparents,:) = pop(ind(1:nparents),:);

topparents = pop(ind(1:nparents),:);

% Step 2. Iterate until a new population of the same size is generated. Using the above
% example, you need to iterate 80 times. In each iteration create a new
% offspring chromosome from two randomly selected parent chromosomes. Use
% the function getOffSpring to generate a new offspring.

for j = nparents+1:size(pop,1)    
    randparents = randperm(nparents);    
    newpop(j,:) = getOffSpring(topparents(randparents(1),:),topparents(randparents(2),:),mutateprob);    
end
end

function offspring = getOffSpring(parent1,parent2,mutateprob)
    parentlength = length(parent1);
    split = floor(rand(1,1)*parentlength+1);
    if split == 0
        offspring = parent1;
    elseif split == 6
        offspring = parent2;
    else
        offspring = [parent1(1:split), parent2(split+1:end)];
    end
    
    for i = 1:length(offspring)
        if rand(1,1) < mutateprob
            offspring(i) = abs(offspring(i) -1);
        end
    end
            
end



function score = getScore(chromosome,train_feats,labels)
% Compute the fitness score using 2-fold cross validation and KNN
% classifier

cv = cvpartition(labels,'Kfold',2);
for i = 1:cv.NumTestSets        
    knn = fitcknn(train_feats(cv.training(i),chromosome),labels(cv.training(i)));
    c = predict(knn,train_feats(cv.test(i),chromosome));
    acc(i) = sum(c == labels(cv.test(i)))/numel(c);
end
meanacc = mean(acc);
score = 1*10^4*meanacc + 0.4*sum(chromosome == 0);

% WRITE CODE TO CALCULATE THE SCORE FOR THE GIVEN CHROMOSOME
% BASED ON THE TRAINING ACCURACY MEANACC AND THE NUMBER OF ZEROS IN THE
% CHROMOSOME
end

function pop = generateInitialPopulation(n,ndim)
    pop = [];
    for i = 1:n
        chromosome = randi([0, 1], [1, ndim]);
        pop = [pop;chromosome];
    end
end
