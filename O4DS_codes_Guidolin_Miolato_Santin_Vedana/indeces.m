success = 0;
n = 10000;
good = [];

for i = 1:n
out2 = elu(w2*images(:,i)+b2);
out3 = elu(w3*out2+b3);
out = elu(w4*out3+b4);
big = 0;
[tilde, num] = max(out);
if labels(i) == num - 1
    good = [good i];
    success = success + 1;
end
    

end

rng(2);
index = randsample(good,100);




out2 = @(x) elu(w2*x+b2);
    out3 = @(x) elu(w3*out2(x)+b3);
    model = @(x) elu(w4*out3(x)+b4);