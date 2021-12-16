difference_zo2_1 = [];
for i = ind_zo2_1
    difference_zo2_1 = [difference_zo2_1 new_images_zo2_1(:,find(index==i))-images(:,i)];
end

l1_zo2_1 = [];
for i = 1:length(ind_zo2_1)
    l1_zo2_1 = [l1_zo2_1 norm(difference_zo2_1(:, i), 1)];
end

mean(l1_zo2_1)