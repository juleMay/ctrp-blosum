import numpy as np
from sklearn.cluster import AffinityPropagation, DBSCAN, SpectralClustering
import pandas as pd
import blosum as bl
import time
import math
import os.path
import joblib
import pathlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn import metrics
from pathlib import Path
from scipy.sparse import lil_matrix

class Options():
    PROTEOM='PLASMODIUM'
    #PROTEOM='UP000030673'
    #PROTEOM='ALZHEIMER'
    #PROTEOM='UP000001450'
    REPEAT_PROTEIN_PATH='input_data/' + PROTEOM
    REPEAT_PROTEIN_NAME='repeat_protein'
    VERSION='v7'
    LEVEL='level1'
    REPEAT_PROTEIN_FILE=REPEAT_PROTEIN_PATH + '/' + REPEAT_PROTEIN_NAME + '_' + PROTEOM + '_' + LEVEL + '.csv'
    TOLERANCE=0.96
    NORMALIZE=True # Normalize metric
    DIVIDE_BY_LENGTH=False
    MATRIX_PATH='precalculated_distance_matrix/' + PROTEOM + '/' + VERSION + '/' + LEVEL
    MATRIX_NAME='lev_matrix_blosum'
    RESULT_PATH='results/' + PROTEOM + '/' + VERSION + '/' + LEVEL
    RESULT_AFFINITY_NAME='affinity_propagation_BLOSUM62'
    RESULT_AFFINITY_ID='AffProp_BL62_Cluster'

    RESULT_PATH_FILE=RESULT_PATH + '/' + PROTEOM + '_' + RESULT_AFFINITY_NAME + '_' + VERSION + '_' + LEVEL + '.csv'

    COST_BLOSUM_POSITIVE=0
    COST_BLOSUM_NEGATIVE=3
    COST_BLOSUM_INSERT=4
    COST_BLOSUM_DELETION=4
    MAX_INSERTS=0
    MAX_SUBSTITUTIONS=3
    MAX_DELETIONS=0
    RELATIVE_COST=True

    PREFERENCE=-0.4
    MIN_CLUSTER_SIZE=34
    #MIN_CLUSTER_SIZE=20
    SAMPLE_SIZE=10
    #SAMPLE_SIZE=5
    PLOT_PATH=RESULT_PATH + '/' + PROTEOM + '_' + RESULT_AFFINITY_NAME + '_' + VERSION + '_' + LEVEL + '.png'

class FileUtil():
    def load_csv_file(self, path=Options().REPEAT_PROTEIN_FILE):
        return pd.read_csv(path)
    
class DataMappingUtil():
    def __init__(self, proteins):
        self.PROTEINS = proteins
    
    def get_repeats(self, level = 'level1'):
        unique_proteins = self.PROTEINS.drop_duplicates(subset='consensus_sequence_alignment', keep='first')
        unique_proteins['duplicated_indices'] = unique_proteins.apply(self.get_duplicates, axis=1).tolist()
        if level == 'level1':
            unique_proteins = unique_proteins[['consensus_sequence_alignment', 'id_protein', 'first_residu_involved', 'last_residu_involved', 'duplicated_indices']]
        elif level == 'level2':
            unique_proteins = unique_proteins[['consensus_sequence_alignment', 'id_protein', 'first_residu_involved', 'last_residu_involved', 'id_cluster_level1', 'consensus_sequence_alignment_level1', 'duplicated_indices']]
        elif level == 'level3':
            unique_proteins = unique_proteins[['consensus_sequence_alignment', 'id_protein', 'first_residu_involved', 'last_residu_involved', 'id_cluster_level1', 'consensus_sequence_alignment_level1', 'id_cluster_level2', 'consensus_sequence_alignment_level2', 'duplicated_indices']]

            unique_proteins['cluster_center_consensus_sequence_alignment'] = 1
            unique_proteins['id_cluster'] = 1
            unique_proteins['number_data_points'] = 1
            unique_proteins['cluster_data_point'] = 1
            unique_proteins['is_cluster_center'] = 1
            unique_proteins['distance_to_center'] = 1
        return unique_proteins
    
    def get_duplicates(self, row):
        return self.PROTEINS[self.PROTEINS['consensus_sequence_alignment'] == row['consensus_sequence_alignment']].index.tolist()
    
class PlotUtil():
    def plot_affinity_cluster_tsne(self, n_clusters, labels, cluster_centers_indices, X, words, silhouette = 0, min_cluster_size = Options().MIN_CLUSTER_SIZE, sample_size = Options().SAMPLE_SIZE):
        # Step 1: Apply MDS to reduce the dimensionality of the distance matrix X
        tsne = TSNE(n_components=2, metric='precomputed', init='random', random_state=69, n_jobs=-1, verbose=True)
        distance_matrix = X  # Calculate pairwise distance matrix
        X_2d = tsne.fit_transform(distance_matrix)  # Apply MDS to reduce to 2D

        # Step 2: Plot the clusters in 2D space using MDS results
        plt.close("all")
        plt.figure(figsize=(10, 7))
        plt.figure(1)
        plt.clf()
        fig, ax = plt.subplots()
        fig.set_size_inches(10, 7)
        colors = plt.cycler("color", plt.cm.viridis(np.linspace(0, 1, n_clusters)))
        for k, col in zip(range(n_clusters), colors):
            class_members = labels == k
            if len(X_2d[class_members, 0]) > min_cluster_size:
                #print("class member")
                #print(class_members)
                cluster_center = X_2d[cluster_centers_indices[k]]  # Use MDS-reduced center

                indices = np.random.choice(X_2d[class_members].shape[0], size=sample_size, replace=False)
                subsample = np.array([X_2d[class_members][i].tolist() for i in indices])
                # Plot points in the cluster
                ax.scatter(
                    subsample[:,0], subsample[:,1], marker=".", color=col["color"], alpha=0.5
                )

                # Plot the center of the cluster
                ax.scatter(
                    cluster_center[0], cluster_center[1], s=14, color="black", marker="x",
                )

                plt.annotate(text=words[cluster_centers_indices[k]], xy=(cluster_center[0], cluster_center[1]),
                            xytext=(15, -15), textcoords='offset points', arrowprops=dict(arrowstyle='->'))

                # Plot lines connecting points to their cluster center
                for x in subsample:
                    ax.plot(
                        [cluster_center[0], x[0]], [cluster_center[1], x[1]], color=col["color"], alpha=0.5
                    )

        # Set title and display the plot
        plt.title('')
        plt.title(f"t-SNE of Similarity Matrix\nEstimated number of clusters: {n_clusters}\nSilhouette Coefficient: {silhouette}")
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.savefig(Options().PLOT_PATH)

    def print_cluster(self, cluster_dictionary, words, repeats, proteins, code = '', cluster_id_count = 1):
        num_clusters = len(cluster_dictionary)
        count = cluster_id_count
        result = pd.DataFrame([])

        for repeat, cluster in cluster_dictionary.items():
            center_key = words[repeat]
            for point in cluster:
                flag = True
                try:
                    flag = int(point[0]) > -1
                except ValueError:
                    flag = False
                if flag:
                    point_index = int(point[0])
                    repeat_key = words[point_index]
                    data_point = repeats.loc[repeats['consensus_sequence_alignment'] == repeat_key]
                    data_point['is_cluster_center'] = repeat == point_index
                    if len(data_point['duplicated_indices'].tolist()[0]) > 1:
                        duplicates = proteins.iloc[data_point['duplicated_indices'].tolist()[0][1::]][['consensus_sequence_alignment', 'id_protein', 'first_residu_involved', 'last_residu_involved']]
                        duplicates['is_cluster_center'] = False
                        data_point = pd.concat([data_point, duplicates], ignore_index=True)
                    data_point['cluster_center_consensus_sequence_alignment'] = center_key
                    data_point['id_cluster'] = code + str(count).zfill(len(str(num_clusters)))
                    data_point['number_data_points'] = len(cluster)
                    data_point['cluster_data_point'] = point[1]
                    data_point['distance_to_center'] = point[2]
                    result = pd.concat([result, data_point])
            count += 1
        result = result.drop(columns=['duplicated_indices'])
        return result, count

class ClusteringUtil():
    def affinity_model(self, lev_similarity, damping_factor=0.5, n_itetations=10000, preference=None):
      """ precomputed es que va a usar la matriz de similitud ya calculada, valor de amotiguamiento es un factor de convergencia para el modelo  """
      affprop = AffinityPropagation(affinity="precomputed", damping=damping_factor, max_iter=n_itetations, verbose=True, convergence_iter=15, preference=preference)
      """ entrena el modelo """
      affprop.fit(lev_similarity)
      return affprop

    def dbscan_model(self, max_distance, min_density, matrix):
      """ precomputed es que va a usar la matriz de similitud ya calculada """
      X = matrix
      db = DBSCAN(eps=max_distance, min_samples=min_density, metric='precomputed')
      """ entrena el modelo """
      db.fit(X)
      return db

    def spectral_model(self, lev_similarity, num_clusters):
      """ precomputed es que va a usar la matriz de similitud ya calculada """
      spect = SpectralClustering(affinity="precomputed", n_clusters=num_clusters)
      """ entrena el modelo """
      spect.fit_predict(X=lev_similarity)
      return spect

    def get_clusters(self, model, words, matrix=[], show_plot=True):
      cluster_dictionary = {}
      if isinstance(model, DBSCAN):
        core_samples_mask = np.zeros_like(model.labels_, dtype=bool)
        core_samples_mask[model.core_sample_indices_] = True
      """ itera sobre las etiquetas unicas de los grupos encontrados """
      for cluster_id in np.unique(model.labels_):
          n_clusters = 0
          if (cluster_id >= 0):
            """ encuentra la palabra en el centro del grupo """
            if isinstance(model, AffinityPropagation):
              cluster_center_index = model.cluster_centers_indices_[cluster_id]
              exemplar = words[cluster_center_index]
              labels = model.labels_
              n_clusters = len(np.unique(labels)) - (1 if -1 in labels else 0)
            elif isinstance(model, DBSCAN):
              core_points_indices = np.where(core_samples_mask & (model.labels_ == cluster_id))[0]
              cluster_center_index = core_points_indices[0]
              exemplar = words[cluster_center_index]
              labels = model.labels_
              n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            elif isinstance(model, SpectralClustering):
              cluster_center_index = np.where(model.labels_ == cluster_id)[0][0]
              exemplar = words[cluster_center_index]
              labels = model.labels_
              n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

            """ busca cada palabra unica que pertence al grupo """
            cluster_indices = np.unique(np.nonzero(model.labels_==cluster_id))
            cluster = [[point,words[point],matrix[cluster_center_index][point]] for point in cluster_indices]
            cluster_dictionary.update({cluster_center_index: cluster})

      silhouette=-1.0
      print("Estimated number of clusters: %d" % n_clusters)
      try:
        silhouette=metrics.silhouette_score(matrix, labels, metric="precomputed")
        print("Silhouette Coefficient: %0.3f" % silhouette)
      except Exception:
         print("Silhouette Calculation Failed")

      if show_plot and n_clusters > 1:
        if isinstance(model, AffinityPropagation):
          PlotUtil().plot_affinity_cluster_tsne(n_clusters, model.labels_, model.cluster_centers_indices_, matrix, words, silhouette)
        elif isinstance(model, DBSCAN):
          PlotUtil().plot_affinity_cluster_tsne(n_clusters, model.labels_, core_points_indices, matrix, words, silhouette)
        elif isinstance(model, SpectralClustering):
          PlotUtil().plot_affinity_cluster_tsne(n_clusters, model.labels_, np.where(model.labels_ == cluster_id), matrix, words, silhouette)
      return cluster_dictionary

    def calculate_lev(self, words, distance_func, upper_matrix=1):
        num_words = len(words)

        # Initialize the similarity matrix
        # lev_similarity = np.zeros((num_words, num_words), dtype=np.float32)
        lev_similarity = lil_matrix((num_words, num_words), dtype=np.float32)

        start_time = time.time()
        # Compute distances
        for i in range(num_words):
            for j in range(i+upper_matrix, num_words):
                if i != j:
                    # Calculate distance between words i and j
                    dist = distance_func(words[i], words[j])
                    lev_similarity[i, j] = lev_similarity[j, i] = -dist

            elapsed_time = time.time() - start_time
            print(f"Progress: {i + 1}/{num_words} words processed. Elapsed time: {elapsed_time:.2f} seconds")

        return lev_similarity.toarray()

class DistanceMatrixUtil():
    def __init__(self):
        self.BLOSUM_MATRIX = bl.BLOSUM(62,5)
    
    def calc_blosum_distance(self, c1, c2):
        substitution_prob = self.BLOSUM_MATRIX[c1][c2]
        if substitution_prob > 1:
            return Options().COST_BLOSUM_POSITIVE
        else:
            return Options().COST_BLOSUM_NEGATIVE

    def _edit_dist_init(self, len1, len2):
        lev = []
        for i in range(len1):
            lev.append([0] * len2)  # initialize 2D array to zero
        for i in range(len1):
            lev[i][0] = i           # column 0: 0,1,2,3,4,...
        for j in range(len2):
            lev[0][j] = j           # row 0: 0,1,2,3,4,...
        return lev

    def _edit_dist_step(self, lev, i, j, s1, s2, cost_subs=0):
        c1 = s1[i - 1]
        c2 = s2[j - 1]

        if c1 == c2:
            lev[i][j] = lev[i - 1][j - 1]
            return 0

        len1 = len(s1)
        len2 = len(s2)

        if len1 == len2:
           lev[i][j] = lev[i - 1][j - 1] + self.calc_blosum_distance(c1, c2) + cost_subs
           return 1

        # skipping a character in s1
        a = lev[i - 1][j] + Options().COST_BLOSUM_INSERT
        print()
        # skipping a character in s2
        b = lev[i][j - 1] + Options().COST_BLOSUM_DELETION
        # substitution
        c = lev[i - 1][j - 1] + self.calc_blosum_distance(c1, c2) + cost_subs # (c1 != c2)

        # pick the cheapest
        lev[i][j] = min(a, b, c)

        if c < min(a, b):
            return 1
        return 0


    def blosum_distance(self, s1, s2):
        """
        Calculate the Levenshtein edit-distance between two strings.
        The edit distance is the number of characters that need to be
        substituted, inserted, or deleted, to transform s1 into s2.  For
        example, transforming "rain" to "shine" requires three steps,
        consisting of two substitutions and one insertion:
        "rain" -> "sain" -> "shin" -> "shine".  These operations could have
        been done in other orders, but at least three steps are needed.

        This also optionally allows transposition edits (e.g., "ab" -> "ba"),
        though this is disabled by default.

        :param s1, s2: The strings to be analysed
        :param transpositions: Whether to allow transposition edits
        :type s1: str
        :type s2: str
        :type transpositions: bool
        :rtype int
        """
        # set up a 2-D array
        len1 = len(s1)
        len2 = len(s2)

        if not Options().NORMALIZE:
            if len1 - len2 > Options().MAX_DELETIONS:
                return 1000 * abs(len1 - len2)

            if len2 - len1 > Options().MAX_INSERTS:
                return 1000 * abs(len2 - len1)
        elif len1 - len2 > Options().MAX_DELETIONS or len2 - len1 > Options().MAX_INSERTS:
            return 1.0

        max_subs = Options().MAX_SUBSTITUTIONS

        if Options().RELATIVE_COST:
            max_subs = math.ceil(len1/Options().MAX_SUBSTITUTIONS)

        lev = self._edit_dist_init(len1 + 1, len2 + 1)


        # iterate over the array
        num_subs = 0
        for i in range(len1):
            for j in range(len2):
                num_subs += self._edit_dist_step(lev, i + 1, j + 1, s1, s2, 0 if num_subs < max_subs else 1)

        result = lev[len1][len2]/(max(len1,len2)*4)
        if Options().NORMALIZE:
            return result
        else:
            return lev[len1][len2]

    
    def get_repeats(self, len_limit, repeat_protein_dictionary):
        repeats_df = repeat_protein_dictionary[len_limit][['consensus_sequence_alignment', 'id_protein', 'first_residu_involved', 'last_residu_involved']]
        # repeats_df.sort()
        repeats_train = np.asarray(repeats_df['consensus_sequence_alignment'].tolist())
        repeats_indices = {i: row for i, (_, row) in enumerate(repeats_df.iterrows())}
        return repeats_train, repeats_indices
    
    def calculate_matrix(self, words, file_name, path = Options().MATRIX_PATH):
        if (os.path.isfile(path + '/' + file_name)):
            matrix = joblib.load(path + '/' + file_name)
        else:
            Path(path).mkdir(parents=True, exist_ok=True)
            matrix = ClusteringUtil().calculate_lev(words, self.blosum_distance)
            joblib.dump(matrix, path + '/' + file_name)
        return matrix
    
    def get_matrix(self, repeats, matrix_name = Options().MATRIX_NAME):
        words = np.asarray(repeats['consensus_sequence_alignment'].tolist())
        matrix = self.calculate_matrix(words, matrix_name + '_length_' + str(0))

        return words, matrix

def main():
    cluster_id_count = 1
    Path(Options().RESULT_PATH).mkdir(parents=True, exist_ok=True)
    proteins = FileUtil().load_csv_file(Options().REPEAT_PROTEIN_FILE)
    repeats = DataMappingUtil(proteins).get_repeats()
    words, matrix = DistanceMatrixUtil().get_matrix(repeats)

    cluster_dictionary = ClusteringUtil().get_clusters(ClusteringUtil().affinity_model(matrix, damping_factor=Options().TOLERANCE, n_itetations=100000000, preference=Options().PREFERENCE), words, -1.0*matrix)
    result, cluster_id_count = PlotUtil().print_cluster(cluster_dictionary, words, repeats, proteins, Options().PROTEOM + '_' + Options().RESULT_AFFINITY_ID + '_', cluster_id_count)
    csvfile = pathlib.Path(Options().RESULT_PATH_FILE)
    result.to_csv(csvfile, index=False, mode='w')

if __name__ == "__main__":
    main()