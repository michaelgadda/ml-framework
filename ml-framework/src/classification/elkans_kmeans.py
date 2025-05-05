class ElkanKMeans:
    def __init__(self, k_clusters: int = 3, epochs: int=100, eps:float = .1, init_indices = None):
        self.k = k_clusters
        self.epochs = epochs
        self.eps = eps
        self.centroids = None
        self.clusters = None
        self.init_indices = init_indices
        self.distance_calculation_tracker = 0
        self.time_taken = 0
        self.centroid_progression = []

    #Deprecated
    def _get_points_to_be_checked_old(self, X_centroids, upper_bounds, centroid_distances, lower_bounds, k):
        cluster = np.where(X_centroids == k)[0]
        points_to_be_checked_idxs = self._get_points_to_be_checked_indices(cluster, upper_bounds, centroid_distances, k)
        c_points_to_be_checked, c_centroids_to_be_checked = self._get_points_and_centroids_to_be_checked_centroid_distances(upper_bounds, centroid_distances, k, points_to_be_checked_idxs)
        points_to_be_checked, centroids_to_be_checked = self._get_points_and_centroids_to_be_checked_old(lower_bounds, points_to_be_checked_idxs, upper_bounds)
        points_to_be_checked = points_to_be_checked_idxs[points_to_be_checked]
        points_that_need_to_be_checked, centroids_that_need_to_be_checked = self.merge_points_that_need_to_be_checked(c_centroids_to_be_checked, c_points_to_be_checked, centroids_to_be_checked, points_to_be_checked )
        return points_that_need_to_be_checked, centroids_that_need_to_be_checked


    #Deprecated
    def _combine_points_that_need_to_be_checked(self, c_centroids_to_be_checked, c_points_to_be_checked, centroids_to_be_checked, points_to_be_checked ):
        points_that_need_to_be_checked = copy.deepcopy(points_to_be_checked)
        centroids_that_need_to_be_checked = copy.deepcopy(centroids_to_be_checked)
        for c_index, c_point in enumerate(c_points_to_be_checked):
            for index, point in enumerate(points_to_be_checked):
                if c_point == point and c_centroids_to_be_checked[c_index] == centroids_to_be_checked[index]:
                    break
                elif index == len(points_that_need_to_be_checked)-1:
                    points_that_need_to_be_checked = np.append(points_that_need_to_be_checked, c_point)
                    centroids_that_need_to_be_checked = np.append(centroids_that_need_to_be_checked, c_centroids_to_be_checked[c_index])
                    break
        return points_that_need_to_be_checked, centroids_that_need_to_be_checked


    #Deprecated
    def _merge_points_that_need_to_be_checked(self, c_centroids_to_be_checked, c_points_to_be_checked, centroids_to_be_checked, points_to_be_checked ):
        points_that_need_to_be_checked = []
        centroids_that_need_to_be_checked = []
        for c_index, c_point in enumerate(c_points_to_be_checked):
            for index, point in enumerate(points_to_be_checked):
                if c_point == point and c_centroids_to_be_checked[c_index] == centroids_to_be_checked[index]:
                    points_that_need_to_be_checked.append(c_point)
                    centroids_that_need_to_be_checked.append(c_centroids_to_be_checked[c_index])
                    break
        return np.array(points_that_need_to_be_checked, dtype=int), np.array(centroids_that_need_to_be_checked, dtype=int)

    #Deprecated
    def _get_points_to_be_checked_v2(self, X_centroids, upper_bounds, centroid_distances, lower_bounds, k):
        cluster = np.where(X_centroids == k)[0]
        #checking if a points distance is less than 1/2 any other centroids distance
        points_to_be_checked_idxs = self._get_points_to_be_checked_indices(cluster, upper_bounds, centroid_distances, k)
        points_to_be_checked_idxs = points_to_be_checked_idxs.astype(int)
        # Checking what centroids have a distance that is less than 1/2 the current centroid to point distance
        c_points_to_be_checked, c_centroids_to_be_checked = self._get_points_and_centroids_to_be_checked_centroid_distances(upper_bounds, centroid_distances, k, points_to_be_checked_idxs)
        points_to_be_checked_stat, centroids_to_be_checked  = self._get_points_and_centroids_to_be_checked_old(lower_bounds, points_to_be_checked_idxs, upper_bounds)
        points_to_be_checked = points_to_be_checked_idxs[points_to_be_checked_stat]
        c_recalculate = np.full((lower_bounds.shape[0], self.k), False)
        q_recalculate = np.full((lower_bounds.shape[0], self.k), False)
        q_recalculate[points_to_be_checked.astype(int), centroids_to_be_checked.astype(int)] = True
        c_recalculate[c_points_to_be_checked.astype(int), c_centroids_to_be_checked.astype(int)] = True
        points_that_need_to_be_computed, centroids_that_need_to_be_computed = np.where(np.logical_and(c_recalculate == True, q_recalculate == True))
        return points_that_need_to_be_computed, centroids_that_need_to_be_computed


    def _euclidean_distance(self, x1: np.array, x2: np.array, axis=0) -> np.array:
        d = np.sqrt(np.sum((x1-x2)**2, axis = axis, dtype=np.float64), dtype=np.float64)
        if isinstance(d, float):
            self.distance_calculation_tracker += 1
        else:
            n_d = np.array(d)
            if len(n_d.shape) == 1:
                n_d = n_d.reshape(-1,1)
            self.distance_calculation_tracker += n_d.shape[0] * n_d.shape[1]
        return d


    def _get_distances_between_centroids(self, centroids: np.ndarray) -> list[list[float]]:
        centroid_distances = [[] for x in range(self.k)]
        centroid_idx = 0
        while centroid_idx < self.k:
            comparative_centroid_idx = centroid_idx + 1
            while comparative_centroid_idx < self.k:
                temp_d = self._euclidean_distance(centroids[centroid_idx], centroids[comparative_centroid_idx])
                centroid_distances[centroid_idx].append([comparative_centroid_idx, temp_d])
                centroid_distances[comparative_centroid_idx].append([centroid_idx, temp_d])
                comparative_centroid_idx += 1
            centroid_idx += 1
        centroid_distances = [sorted(row, key=lambda x: x[1]) for row in centroid_distances]
        return centroid_distances


    def _get_adjusted_centroids(self, X: np.ndarray, X_centroids: np.array) -> np.array:
        adjusted_centroids = []
        for k in range(self.k):
            cluster = np.where(np.array(X_centroids) == k)[0]
            adjusted_centroid = np.sum(X[cluster], axis=0, dtype=np.float64)/cluster.shape[0]
            adjusted_centroids.append(adjusted_centroid)
        return np.array(adjusted_centroids)


    def _get_distance_between_prev_and_curr_centroids(self, curr_centroids: np.array, prev_centroids: np.array) -> list[float]:
        curr_prev_centroid_dis = []
        for k in range(self.k):
            temp_d = self._euclidean_distance(curr_centroids[k], prev_centroids[k])
            curr_prev_centroid_dis.append(temp_d)
        return curr_prev_centroid_dis


    def _get_points_to_be_checked_indices(self, cluster, upper_bounds, centroid_distances, k):
        return cluster[np.where(upper_bounds[cluster] >=  centroid_distances[k][0][1]*.5)]


    def _get_points_and_centroids_to_be_checked_old(self, lower_bounds, points_to_be_checked_idxs, upper_bounds):
        return np.where(lower_bounds[points_to_be_checked_idxs, :] < upper_bounds[points_to_be_checked_idxs].reshape(-1,1))


    def _get_points_and_centroids_to_be_checked(self, lower_bounds, points_to_be_checked_idxs, centroids_to_be_checked_idxs, upper_bounds):
        return np.where(lower_bounds[points_to_be_checked_idxs.astype(int), centroids_to_be_checked_idxs.astype(int)] < upper_bounds[points_to_be_checked_idxs.astype(int)])


    def _get_points_and_centroids_to_be_checked_centroid_distances(self, distance_array, centroid_distances, k, points_to_be_checked_idxs, centroid_indices=None):
        points_to_be_checked = np.array([])
        centroids_to_be_checked = np.array([])
        for centroid_distance in centroid_distances[k]:
            if centroid_indices is not None:
                if centroid_distance[0] in centroid_indices:
                    temp_pts_to_be_checked_idxs = points_to_be_checked_idxs[np.where(centroid_indices == centroid_distance[0])]
                    temp_pts_to_be_checked = temp_pts_to_be_checked_idxs[np.where(distance_array[temp_pts_to_be_checked_idxs] > centroid_distance[1]*.5)]
                    points_to_be_checked = np.concatenate((points_to_be_checked, temp_pts_to_be_checked))
                    centroids_to_be_checked = np.concatenate((centroids_to_be_checked, np.full(len(temp_pts_to_be_checked), centroid_distance[0])))
                    continue
                else:
                    continue
            temp_pts_to_be_checked = points_to_be_checked_idxs[np.where(distance_array[points_to_be_checked_idxs] > centroid_distance[1]*.5)]
            points_to_be_checked = np.concatenate((points_to_be_checked, temp_pts_to_be_checked))
            centroids_to_be_checked = np.concatenate((centroids_to_be_checked, np.full(len(temp_pts_to_be_checked), centroid_distance[0])))
        return points_to_be_checked, centroids_to_be_checked



    def _get_points_to_be_checked(self, X_centroids, upper_bounds, centroid_distances, lower_bounds, k):
        cluster = np.where(X_centroids == k)[0]
        points_to_be_checked_idxs = self._get_points_to_be_checked_indices(cluster, upper_bounds, centroid_distances, k)
        c_points_to_be_checked, c_centroids_to_be_checked = self._get_points_and_centroids_to_be_checked_centroid_distances(upper_bounds, centroid_distances, k, points_to_be_checked_idxs)
        points_to_be_checked_stat  = self._get_points_and_centroids_to_be_checked(lower_bounds, c_points_to_be_checked, c_centroids_to_be_checked, upper_bounds)
        points_to_be_checked = c_points_to_be_checked[points_to_be_checked_stat]
        centroids_to_be_checked = c_centroids_to_be_checked[points_to_be_checked_stat]
        return points_to_be_checked, centroids_to_be_checked

    def _calculate_fronebus_relative_error(self, x1, x2):
        x1_norm = np.sqrt(np.sum((x1)**2, dtype=np.float64), dtype=np.float64)
        x2_norm = np.sqrt(np.sum((x2)**2, dtype=np.float64), dtype=np.float64)
        rel_err = np.abs((x1_norm-x2_norm)/x2_norm)
        return rel_err


    def sort_into_clusters(self):
        divided_clusters = []
        for k in range(self.k):
            divided_clusters.append(np.where(self.clusters==k))
        return divided_clusters


    def brute_force_clusters(self, X: np.ndarray, centroids: list) -> list[list]:
      assigned_centroids = np.zeros(X.shape[0])
      for x_index, x in enumerate(X):
        min_distance = None
        for c_index, centroid in enumerate(centroids):
          d = self._euclidean_distance(np.array(x), np.array(centroid))
          if min_distance == None:
            min_distance = d
            assigned_centroids[x_index] = c_index
          elif d < min_distance:
            min_distance = d
            assigned_centroids[x_index] = c_index
      clusters = [np.where(assigned_centroids == index)[0].tolist() for index in range(self.k)]
      return assigned_centroids


    def find_clusters(self, X, centroid_distances, centroids):
        n = X.shape[0]
        X_centroids = np.zeros(n).astype(int)
        X_distances = np.zeros((n,self.k), dtype=np.float64)
        lower_bounds = np.zeros((n,self.k), dtype=np.float64)
        upper_bounds = np.full(n, np.inf, dtype=np.float64)
        for x_index, x in enumerate(X):
            X_distances[x_index, X_centroids[x_index]] = self._euclidean_distance(x, centroids[X_centroids[x_index]])
            upper_bounds[x_index] = X_distances[x_index, X_centroids[x_index]]
            lower_bounds[x_index, X_centroids[x_index]] = X_distances[x_index, X_centroids[x_index]]
            if centroid_distances[X_centroids[x_index]][0][1]*.5 > X_distances[x_index, X_centroids[x_index]]:
                pass
            else:
                k = 0
                while k < self.k-1:
                    if centroid_distances[X_centroids[x_index]][k][1] * .5 > X_distances[x_index, X_centroids[x_index]]:
                        k += 1
                    else:
                        b_distance = self._euclidean_distance(x, centroids[centroid_distances[X_centroids[x_index]][k][0]])
                        lower_bounds[x_index, centroid_distances[X_centroids[x_index]][k][0]] = b_distance
                        X_distances[x_index, centroid_distances[X_centroids[x_index]][k][0]] = b_distance
                        if b_distance < X_distances[x_index, X_centroids[x_index]]:
                            X_centroids[x_index] = centroid_distances[X_centroids[x_index]][k][0]
                            X_distances[x_index, centroid_distances[X_centroids[x_index]][k][0]] = b_distance
                            upper_bounds[x_index] = b_distance
                            k = 0
                        else:
                            k += 1
        return X_centroids, X_distances, lower_bounds, upper_bounds


    def fit(self, X):
        complete_start_time = time.time()
        n = X.shape[0]
        #init clusters
        if self.init_indices == None or len(self.init_indices) != self.k:
            centroids = X[np.random.randint(0, n, self.k), :]
        else:
            centroids = X[self.init_indices]
        r_x = np.array([True]*n)
        self.centroid_progression.append(centroids)
        centroid_distances = self._get_distances_between_centroids(centroids)
        # Finding initial cluster of points respective to initial indices
        X_centroids, X_distances, lower_bounds, upper_bounds = self.find_clusters(X, centroid_distances, centroids=centroids)
        self.clusters = X_centroids
        self.centroids = centroids
        # Getting adjusted initialized centroids ~ correct?
        new_centroids = self._get_adjusted_centroids(X, X_centroids)
        self.centroid_progression.append(new_centroids)
        curr_prev_centroid_dis = self._get_distance_between_prev_and_curr_centroids(centroids, new_centroids)
        centroids = new_centroids
        self.centroids = centroids
        # Getting our new adjusted lower bounds
        lower_bounds = np.maximum(lower_bounds - curr_prev_centroid_dis, 0)
        # Getting our new adjusted upperbounds
        for k in range(self.k):
            upper_bounds[X_centroids == k] += curr_prev_centroid_dis[k]
            r_x[X_centroids == k] = True
        complete_start_time = time.time()
        for epoch in range(self.epochs):
            points_changed = 0
            start_time = time.time()
            # Each time we update our centroids (end of each epoch) we need to upate the distances between them
            centroid_distances = self._get_distances_between_centroids(centroids)
            # Iterating over centroids rather than points b/c we can vectorize points
            for k in range(self.k):
                # Getting the initial set of points we need consider changing
                next_pts_to_be_checked, centroids_to_be_checked = self._get_points_to_be_checked(X_centroids, upper_bounds, centroid_distances, lower_bounds, k)
                next_pts_to_be_checked = next_pts_to_be_checked.astype(int)
                centroids_to_be_checked = centroids_to_be_checked.astype(int)
                # We only need to  consider point, centroid combinations where the considered centroid is not the currently assigned centroid.
                next_pts_to_be_checked = next_pts_to_be_checked[np.where(centroids_to_be_checked != k)]
                centroids_to_be_checked = centroids_to_be_checked[np.where(centroids_to_be_checked != k)]
                # Now need to update the current distance between each point and its currently assigned centroid IF r_x[x] == True, if not we can use the upper bound as the upperbound is guaranteed to be respective to the currently assigned centroid.
                unique_points_to_be_checked = np.unique(next_pts_to_be_checked)
                pts_to_calculate_distance_for = unique_points_to_be_checked[np.where(r_x[unique_points_to_be_checked] == True)]
                pts_to_use_upper_bounds_for = unique_points_to_be_checked[np.where(r_x[unique_points_to_be_checked] == False)]
                # Setting our current set of distances equal to the respective upperbounds where R_x == False
                X_distances[pts_to_use_upper_bounds_for, X_centroids[pts_to_use_upper_bounds_for]] = upper_bounds[pts_to_use_upper_bounds_for]
                X_distances[pts_to_calculate_distance_for, X_centroids[pts_to_calculate_distance_for]] = self._euclidean_distance(X[pts_to_calculate_distance_for], centroids[X_centroids[pts_to_calculate_distance_for]], axis=1)
                upper_bounds[pts_to_calculate_distance_for] = X_distances[pts_to_calculate_distance_for, X_centroids[pts_to_calculate_distance_for]]
                # Once we have calculated the distance for a specific centroid/point pair, we are guaranteed it they're associated, so we can set r_x[x] == False.
                r_x[pts_to_calculate_distance_for] = False
                # Whenever we calculate the distance of a point we set the lower bounds equal to the newly calculated points
                lower_bounds[pts_to_calculate_distance_for, X_centroids[pts_to_calculate_distance_for]] = X_distances[pts_to_calculate_distance_for, X_centroids[pts_to_calculate_distance_for]]
                # Now checking where any of the distances between our currently assigned centroid & point are greater than any of the points distances b/w our point and any other centroid's lower bounds.
                points_to_be_checked_unindexed_lb = np.where(X_distances[next_pts_to_be_checked, X_centroids[next_pts_to_be_checked]] > lower_bounds[next_pts_to_be_checked, centroids_to_be_checked])[0]
                points_to_be_checked = next_pts_to_be_checked[points_to_be_checked_unindexed_lb]
                centroids_to_be_checked_lb = centroids_to_be_checked[points_to_be_checked_unindexed_lb]
                # Now checking where the current point centroid combo's distance is greater than .5 the distance of the current centroid to any other centroid.
                points_to_be_checked_c, centroids_to_be_checked_c = self._get_points_and_centroids_to_be_checked_centroid_distances(X_distances[:, k], centroid_distances, k, next_pts_to_be_checked, centroid_indices=centroids_to_be_checked)
                k_start_time = time.time()
                recalculate = np.full((n, self.k), False)
                recalculate[points_to_be_checked.astype(int), centroids_to_be_checked_lb.astype(int)] = True
                recalculate[points_to_be_checked_c.astype(int), centroids_to_be_checked_c.astype(int)] = True
                points_that_need_to_be_computed, centroids_that_need_to_be_computed = np.where(recalculate == True)
                points_that_need_to_be_computed, centroids_that_need_to_be_computed = points_that_need_to_be_computed.astype(int), centroids_that_need_to_be_computed.astype(int)
                # Now that we have found the points that potentially need to be reassigned we need to calculate their distances to each of respective centroids that was found to be a potential.
                X_distances[points_that_need_to_be_computed, centroids_that_need_to_be_computed] = self._euclidean_distance(X[points_that_need_to_be_computed], centroids[centroids_that_need_to_be_computed], axis=1)
                lower_bounds[points_that_need_to_be_computed, centroids_that_need_to_be_computed] = X_distances[points_that_need_to_be_computed, centroids_that_need_to_be_computed]
                # Find where the distance of the newly found distances are less than our current centroid/point pair's distance.
                points_to_be_set = np.where(X_distances[points_that_need_to_be_computed, centroids_that_need_to_be_computed] < X_distances[points_that_need_to_be_computed, X_centroids[points_that_need_to_be_computed]])
                points_to_be_updated = points_that_need_to_be_computed[points_to_be_set]
                centroids_to_be_updated = centroids_that_need_to_be_computed[points_to_be_set]
                points_to_set_full_arr = np.full((n, self.k), np.inf)
                points_to_set_full_arr[points_to_be_updated, centroids_to_be_updated] = X_distances[points_to_be_updated, centroids_to_be_updated]
                centroids_to_be_updated = np.argmin(points_to_set_full_arr, axis=1)
                unique_points_to_be_updated = np.unique(points_to_be_updated)
                unique_centroids_to_be_updated = centroids_to_be_updated[unique_points_to_be_updated]
                # Setting the new centroids
                X_centroids[unique_points_to_be_updated] = unique_centroids_to_be_updated
                upper_bounds[unique_points_to_be_updated] = X_distances[unique_points_to_be_updated, unique_centroids_to_be_updated]
                r_x[unique_points_to_be_updated] = False
                points_changed += unique_points_to_be_updated.shape[0]
            # Getting new mean-adjusted centroids
            new_centroids = self._get_adjusted_centroids(X, X_centroids)
            c_ci_d = self._get_distance_between_prev_and_curr_centroids(centroids, new_centroids)
            # Checking if our tolerance for change was met, if not - stop (psuedo convergance)
            if self._calculate_fronebus_relative_error(centroids, new_centroids) < self.eps:
                centroids = new_centroids
                self.centroids = new_centroids
                break
            centroids = new_centroids
            self.centroids = centroids

            #Testing purposes
            self.centroid_progression.append(centroids)

            lower_bounds = np.maximum(lower_bounds - c_ci_d, 0)
            for k in range(self.k):
                upper_bounds[X_centroids == k] += c_ci_d[k]
                r_x[X_centroids == k] = True

        self.centroids = centroids
        centroid_distances = self._get_distances_between_centroids(centroids)
        #X_centroids, X_distances, lower_bounds, upper_bounds = self.find_clusters(X, centroid_distances, centroids=centroids)
        self.clusters = X_centroids
        end_time = time.time()
        elapsed_time = end_time - complete_start_time
        self.time_taken = elapsed_time
