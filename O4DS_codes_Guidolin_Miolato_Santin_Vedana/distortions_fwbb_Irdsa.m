difference_fwbb = [];
for i = ind_fwbb
    difference_fwbb = [difference_fwbb new_images_fwbb(:,find(index==i))-images(:,i)];
end

l1_fwbb = [];
for i = 1:length(ind_fwbb)
    l1_fwbb = [l1_fwbb norm(difference_fwbb(:, i), 1)];
end

mean(l1_fwbb)