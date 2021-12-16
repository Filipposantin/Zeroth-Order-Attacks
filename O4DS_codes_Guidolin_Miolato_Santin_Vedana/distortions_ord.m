difference_ord = [];
for i = ind_ord
    difference_ord = [difference_ord new_images_ord(:,find(index==i))-images(:,i)];
end

l1_ord = [];
for i = 1:length(ind_ord)
    l1_ord = [l1_ord norm(difference_ord(:, i), 1)];
end

mean(l1_ord)