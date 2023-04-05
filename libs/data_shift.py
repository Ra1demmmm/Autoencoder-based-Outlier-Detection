from sklearn.neighbors import kneighbors_graph, NearestNeighbors
import numpy as np

def mean_shift(data, k, times=1, reserve_all=False):
    n = data.shape[0]
    data = data.copy()
    shifted_data = data.copy()
    if reserve_all:
        shifted_alltimes = [None]*times
    for time in range(times):
        A = kneighbors_graph(data, k, mode='connectivity', include_self=True)
        A.toarray()
        for point in range(n):
            neighbor_list = np.argwhere(A[point] == 1.)[:,1]
            shifted_data[point] = np.mean(data[neighbor_list], axis=0)
        data = shifted_data.copy()
        if reserve_all:
            shifted_alltimes[time] = shifted_data.copy()

    if reserve_all:
        return shifted_alltimes
    else:
        return shifted_data


def mean_shift_samples(data, samples, k, times=1):
    if len(data) != times:
        print('Error! Don\'t have enough data!')
        exit()
    m = samples.shape[0]
    samples = samples.copy()
    shifted_samples = samples.copy()

    for time in range(times):
        neigh = NearestNeighbors(n_neighbors=k - 1)
        neigh.fit(data[time])
        for i in range(m):
            neighbor_list = neigh.kneighbors([samples[i]], return_distance=False)
            neighbor_data = np.concatenate(([samples[i]], data[time][neighbor_list][0]), axis=0)
            shifted_samples[i] = np.mean(neighbor_data, axis=0)

        samples = shifted_samples

    return shifted_samples

