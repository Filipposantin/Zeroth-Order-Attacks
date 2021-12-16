function plot_image(mod,image, model)
    t = tiledlayout(1,3);
    ax1 = nexttile;
    colormap gray
    imagesc(reshape(image, 28,28)')
    ax2 = nexttile;
    colormap gray
    imagesc(reshape(mod, 28,28)')
    ax3 = nexttile;
    colormap gray
    imagesc(reshape(abs(mod - image), 28,28)')
    [~, O_class] = max(model(image));
    [~, N_class] = max(model(mod));
    fprintf(['\nPrevious Class = %-i'...
         '\n New Class = %-i \n\n'], ...
         O_class -1, N_class - 1)
     xticklabels(ax1,{})
     yticklabels(ax1, {})
     xticklabels(ax2,{})
     yticklabels(ax2, {})
     xticklabels(ax3,{})
     yticklabels(ax3, {})
end