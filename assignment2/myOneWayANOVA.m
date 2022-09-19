function F = myOneWayANOVA(IV, DV)
    sum_SS_B = 0;
    sum_SS_W = 0;
    x = mean(IV);
    unique_DV = unique(DV);
    len = length(unique_DV);
    if len == 1 || length(IV)~= length(DV)
        error('F-statistic cannnot be calculated!');
    else
        for i = 1:len
            next_list = IV(DV == unique_DV(i));
            m = mean(next_list);
            partial_SS_B = length(next_list)*((m-x)^2);
            partial_SS_W = (length(next_list)- 1)*var(next_list);
            sum_SS_B = sum_SS_B + partial_SS_B;
            sum_SS_W = sum_SS_W + partial_SS_W;  
        end
        SS_B = sum_SS_B/(len - 1);
        SS_W = sum_SS_W/(length(DV) - len);
        F = SS_B/SS_W;
    end
end