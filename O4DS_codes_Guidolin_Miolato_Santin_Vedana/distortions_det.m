ind_det = matfile('ind_det.mat');
ind_det = ind_det.ind_det;

difference_det = [];
for i = ind_det
    difference_det = [difference_det new_images_det(:,find(index==i))-images(:,i)];
end

l1_det = [];
for i = 1:length(ind_det)
    l1_det = [l1_det norm(difference_det(:, i), 1)];
end

mean(l1_det)