difference_fwbb_1 = [];
for i = ind_fwbb_1
    difference_fwbb_1 = [difference_fwbb_1 new_images_fwbb_1(:,find(index==i))-images(:,i)];
end

l1_fwbb_1 = [];
for i = 1:length(ind_fwbb_1)
    l1_fwbb_1 = [l1_fwbb_1 norm(difference_fwbb_1(:, i), 1)];
end

mean(l1_fwbb_1)