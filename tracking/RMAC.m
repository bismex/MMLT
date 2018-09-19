function vector = RMAC(feature)

feature_area = size(feature,1) * size(feature,2);


min_size = min(size(feature,1), size(feature,2));
max_size = max(size(feature,1), size(feature,2));

i = 1;
vector = zeros(([1 1 size(feature,3)]));
while 1
    sample_size = (2*min_size/(i+1));
    if (sample_size^2)/feature_area < 0.4
       break; 
    end
    sample_size = round(sample_size);
    
%     reduction = max_size - sample_size;
    for j = 1 : i^2 
        row_start = floor(mod(j-1, i)*sample_size/2) + 1;
        col_start = floor(floor((j-1)/i)*sample_size/2) + 1;
        tmp_vector = max(max(feature(col_start:col_start+sample_size-1,...
            row_start:row_start+sample_size-1, :),[],1),[],2);
        vector = vector + tmp_vector;
    end
    i=i+1; 
end

vector = vector./sqrt(sum(vector.^2,3));

end