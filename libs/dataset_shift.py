import os
import numpy as np
from libs.data_shift import mean_shift, mean_shift_samples

def data_mean_shift(root, dataset, train_data, tn, test_data, k_max = 100):
    n = len(train_data)

    for k in range(2, min(n, k_max+1)):
        dump_dir = os.path.join(root, dataset, str(k))

        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)


        [shifted_train_mean_t1, shifted_train_mean_t2, shifted_train_mean_t3] = mean_shift(train_data.numpy(), k, 3, True)

        shifted_valid_mean_t1 = shifted_train_mean_t1[tn:]
        shifted_valid_mean_t2 = shifted_train_mean_t2[tn:]
        shifted_valid_mean_t3 = shifted_train_mean_t3[tn:]

        shifted_test_mean_t1 = mean_shift_samples([train_data.numpy()], test_data.numpy(), k, 1)
        shifted_test_mean_t2 = mean_shift_samples(
            [train_data.numpy(), shifted_train_mean_t1],
            test_data.numpy(), k, 2)
        shifted_test_mean_t3 = mean_shift_samples(
            [train_data.numpy(), shifted_train_mean_t1, shifted_train_mean_t2],
            test_data.numpy(), k, 3)

        np.savetxt(os.path.join(dump_dir, 'shifted_train_mean_t1.txt'), shifted_train_mean_t1)
        np.savetxt(os.path.join(dump_dir, 'shifted_train_mean_t2.txt'), shifted_train_mean_t2)
        np.savetxt(os.path.join(dump_dir, 'shifted_train_mean_t3.txt'), shifted_train_mean_t3)


        np.savetxt(os.path.join(dump_dir, 'shifted_valid_mean_t1.txt'), shifted_valid_mean_t1)
        np.savetxt(os.path.join(dump_dir, 'shifted_valid_mean_t2.txt'), shifted_valid_mean_t2)
        np.savetxt(os.path.join(dump_dir, 'shifted_valid_mean_t3.txt'), shifted_valid_mean_t3)

        np.savetxt(os.path.join(dump_dir, 'shifted_test_mean_t1.txt'), shifted_test_mean_t1)
        np.savetxt(os.path.join(dump_dir, 'shifted_test_mean_t2.txt'), shifted_test_mean_t2)
        np.savetxt(os.path.join(dump_dir, 'shifted_test_mean_t3.txt'), shifted_test_mean_t3)

