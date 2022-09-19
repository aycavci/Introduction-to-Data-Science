function [Fvalue] = myOneWayANOVA(IV,DV)
categories = unique(DV);

if length(IV) ~= length(DV)
    error("IV and DV are not of equal length")
elseif length(categories) <= 1
    error("DV only has 1 unique value")  
else
    grandmean = mean(IV);
    SSB = 0;
    SSW = 0;
    for i = 1:length(categories)
        categoryvalues = IV(DV == categories(i));
        nrvalues = length(categoryvalues);
        groupmean = mean(categoryvalues);
        for j = 1:length(categoryvalues)
            SSW = SSW + (categoryvalues(j) - groupmean)^2;
        end
        SSB = SSB + nrvalues * (groupmean - grandmean)^2;
    end
    MSB = SSB / (length(categories)-1);
    MSW = SSW / (length(IV) - length(categories));
    Fvalue = MSB/MSW;
end
    
end

