difference_zo2 = [];
for i = ind_zo2
    difference_zo2 = [difference_zo2 new_images_zo2(:,find(index==i))-images(:,i)];
end

l1_zo2 = [];
for i = 1:length(ind_zo2)
    l1_zo2 = [l1_zo2 norm(difference_zo2(:, i), 1)];
end

mean(l1_zo2)
