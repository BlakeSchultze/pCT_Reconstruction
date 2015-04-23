close all;

num_histories_2_read = 20;
values_per_history = 15;
sizeA = num_histories_2_read * values_per_history;
simulated_data = fopen('simdata_trans1_000.dat');
A = fread(simulated_data, sizeA, 'float')
B = zeros(num_histories_2_read, values_per_history);
for( i = 1: num_histories_2_read)
   B(i,:) = A( (i-1) * values_per_history + 1: i * values_per_history);
end
B;
A = reshape(A, num_histories_2_read, values_per_history );
A2 = fread(simulated_data, sizeA, 'char')
