%% GA naive Bayes
pop_size=10;
crossover=0.25;
mutation=0.1;
row=size(GlassNew,1);
col=size(GlassNew,2);
dataset1=GlassNew(1:row,1:col-1);
N_data=size(dataset1,2);
dataset=zeros(row,1);
classlabel=GlassNew(1:row,col);
X=ceil(row/10);
accuracy=zeros(10,1);
population=0 + (rand(pop_size,N_data) > 0.4);
fitness=zeros(pop_size,1);
stop=sum(fitness)/10;
max_count=0;
%% fitness evaluation
while( stop~=100 && max_count~=50)
for i=1:pop_size
    for j=1:N_data
        if(population(i,j)==1)
            dataset=cat(2,dataset1(:,j),dataset(:,:));
        end
    end  
    N=size(dataset,2)-1;
    for fold=1 :10
        tenfold=fold*X-(X-1);
        if(fold==10)
            testdata=dataset(tenfold:row,1:N);
            testclass=classlabel(tenfold:row,1);
        else
            testdata=dataset(tenfold:tenfold + (X-1),1:N);
            testclass=classlabel(tenfold:tenfold +(X-1),1);
        end
    
        if (fold==1)
            traindata=dataset(tenfold+X:row,1:N);
            trainclass=classlabel(tenfold+X:row,1);
        else
            traindata=cat(1,dataset(1:tenfold-1,1:N),dataset(tenfold+X:row,1:N));
            trainclass=cat(1,classlabel(1:tenfold-1,1),classlabel(tenfold+X:row,1));
        end
        accuracy=nb(traindata,trainclass,testdata,testclass);
         % NBModel = fitNaiveBayes(traindata,trainclass);
         %predclass=predict(NBModel,testdata);
        %confusionmatrix = confusionmat(testclass,predclass); 
        %accuracy(fold)=(confusionmatrix(1,1)+confusionmatrix(2,2))/sum(sum(confusionmatrix));
    end
    %fitness(i)=sum(accuracy)*10;
    fitness(i)=accuracy;
end
%% selection
prob=fitness/sum(fitness);
cum_prob(1)=prob(1);
for i=2:pop_size
    cum_prob(i)=prob(i)+cum_prob(i-1);
end
random=rand(pop_size,1);
selection=zeros(pop_size,1);
for i=1:pop_size
    flag=0;
    for sel_test=1:pop_size
        if(random(i)<cum_prob(sel_test))
            selection(i)=sel_test;
            flag=1;
        end
        if(flag==1)
            break
        end
    end
end
%% crossover
for itr=1:pop_size
    new_population(itr,:)=population(selection(itr),:);
end
cross_rand=rand(pop_size,1);
itr=1;
for cross_itr=1:10
    if(cross_rand(cross_itr)<crossover)
        crosslist(itr)=cross_itr;
        itr=itr+1;
    end
end
cross_point= randi([0 col-2],1,1);
T=size(crosslist,2);
for cross=1:T
    if (cross==T)
        cross_pop(cross,:)=cat(2,new_population(crosslist(1,cross),1:cross_point),new_population(crosslist(1,1),cross_point+1:size(new_population(crosslist(1,1),:),2)));
    else
    cross_pop(cross,:)=cat(2,new_population(crosslist(1,cross),1:cross_point),new_population(crosslist(1,cross+1),cross_point+1:size(new_population(crosslist(1,cross+1),:),2)));
    end
end
i=1;
for final_cross=1:pop_size
    if(i<size(crosslist,2)&&crosslist(i)==final_cross)
        cross_population(final_cross,:)=cross_pop(i,:);
        i=i+1;
    else
         cross_population(final_cross,:)=new_population(final_cross,:);
    end
end
%% mutation
mutation_rand=randi([1 10],1,1);
mutation_bit=randi([1 col-1],1,1);
cross_population(mutation_rand,mutation_bit)=1-cross_population(mutation_rand,mutation_bit);
population=cross_population;
max_count=max_count+1;
end
