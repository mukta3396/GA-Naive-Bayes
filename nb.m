%X = double(GlassNew(:,1:9));
%Y = double(GlassNew(:,10));
%c = cvpartition(Y,'kfold');
% Create a training set
%x = X(training(c,1),:);
%y = Y(training(c,1));
% test set
%u=X(test(c,1),:);
%v=Y(test(c,1),:);
function [accuracy] = nb(X,XC,Y,YC)
distr='normal';
x=X;
y=XC;
u=Y;
v=YC;
yu=unique(y);
nc=length(yu); % number of classes
ni=size(x,2); % independent variables
ns=length(v); % test set

% compute class probability
for i=1:nc
    fy(i)=sum(double(y==yu(i)))/length(y);
end

switch distr
    case 'normal'
        % normal distribution
        % parameters from training set
        for i=1:nc
            xi=x((y==yu(i)),:);
            mu(i,:)=mean(xi,1);
            sigma(i,:)=std(xi,1);
        end
        % probability for test set
        for j=1:ns
            fu=normcdf(ones(nc,1)*u(j,:),mu,sigma);
            P(j,:)=fy.*prod(fu,2)';
        end
   
    otherwise     
        %disp('invalid distribution stated')
        return

end
% get predicted output for test set
[pv0,id]=max(P,[],2);
for i=1:length(id)
    pv(i,1)=yu(id(i));
end
confMat=confusionmat(v,pv);
%disp(confMat)
conf=sum(pv==v)/length(pv);
accuracy=conf*100;
end
%disp(['accuracy = ',num2str(conf*100),'%'])