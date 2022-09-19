function [Xfull] = MyImpute(Xmissing,S,~)
for i = 1:length(S)
    if S(i) == "Cat"
        Xmode = mode(categorical(Xmissing(:,i)));
        for j = 1:length(Xmissing(:,i))
            if ismissing(Xmissing(j,i))
                Xmissing(j,i) = Xmode;
            end
        end
    end
    if S(i) == "Con"
        Xmean = mean(double(Xmissing(:,i)), 'omitnan');
        for j = 1:length(Xmissing(:,i))
            if isnan(double(Xmissing(j,i)))
                Xmissing(j,i) = Xmean;
            end
        end
    end
Xfull = Xmissing;
end

