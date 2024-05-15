from sklearn.datasets import make_classification
from scipy.stats import pearsonr
from sklearn import svm
import numpy as np
import numpy_indexed as npi
import random
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import pingouin as pg

random.seed(0)
np.random.seed(0)

def generate_data(separation, seed=None):
    
    x, y = make_classification(n_samples=100, n_features=16, n_classes=2,
                               n_informative=16, n_redundant=0, class_sep=separation,
                               flip_y=0., shuffle=False, n_clusters_per_class=1, random_state=seed)
    return x, y


def trainandtest_svm(x, y, shuffle=True, seed=42):
    
    zscore = lambda x: (x - np.mean(x))/np.std(x)
    
    n_train = np.ceil(0.8*len(x)).astype(int)
    
    if shuffle:
        rng = np.random.default_rng(seed=seed)
        shuffled_indices = rng.permutation(len(x))
        x = x[shuffled_indices]
        y = y[shuffled_indices]
    
    train_x = x[:n_train]
    train_y = y[:n_train]
    test_x = x[n_train:]
    test_y = y[n_train:]

    train_z = zscore(train_x)
    test_z = zscore(test_x)

    clf = svm.SVC(kernel='linear')
    clf.fit(train_z, train_y)

    y_pred = clf.predict(test_z)

    # Compute distance from bound
    y = clf.decision_function(test_z)
    w_norm = np.linalg.norm(clf.coef_)
    dist = y / w_norm

    zscoredist = zscore(dist)
    zscoredist[test_y==0] *= -1
    
    return np.mean(zscoredist)


def average_data(list_of_x, list_of_y):
    
    x = np.mean(np.dstack(list_of_x), axis=2)
    y = list_of_y[0]
    
    return x, y


def generate_trials(n_trials=400, n_congruent=300, n_timesteps=10,
                    slope_cong=3.4, interc_cong=0., slope_inc=1.8, interc_inc=0.,
                    max_act=0.3, min_act=0., sigma=0., same_seed=True, seed=123):
    
    assert n_congruent < n_trials
    n_incongruent = n_trials - n_congruent
    
    
    # Congruent trials
    
    congruent_activations = []
    congruent_separations = []
    
    for trial in range(n_congruent):
        activ_sequence = []
        separ_sequence = []
        for t in range(n_timesteps):
            this_act = random.uniform(min_act, max_act)
            this_sep = slope_cong*this_act + interc_cong + np.random.normal(0., sigma)
            activ_sequence.append(this_act)
            separ_sequence.append(this_sep)
        congruent_activations.append(np.hstack(activ_sequence).reshape(1, -1))
        congruent_separations.append(np.hstack(separ_sequence).reshape(1, -1))
        
    congruent_activations = np.vstack(congruent_activations)
    congruent_separations = np.vstack(congruent_separations)
    
    
    # Incongruent trials
    
    incongruent_activations = []
    incongruent_separations = []

    for trial in range(n_incongruent):
        activ_sequence = []
        separ_sequence = []
        for t in range(n_timesteps):
            this_act = random.uniform(min_act, max_act)
            this_sep = slope_inc * this_act + interc_inc + np.random.normal(0., sigma)
            activ_sequence.append(this_act)
            separ_sequence.append(this_sep)
        incongruent_activations.append(np.hstack(activ_sequence).reshape(1, -1))
        incongruent_separations.append(np.hstack(separ_sequence).reshape(1, -1))
        
    incongruent_activations = np.vstack(incongruent_activations)
    incongruent_separations = np.vstack(incongruent_separations)
    
    # Subdivide the generated activations and separations into three splits
    # and generate the actual time sequences of data/classification accuracies
    
    trialsplit = np.array(random.sample([1,2,3]*int(n_congruent/3), n_congruent))
    
    activations_tseries_cong = []
    distances_tseries_cong = []

    #print('Generating congruent...')
    for i in [1, 2, 3]:
        #print(f'Split {i:g}/3')
        thissplit_act = congruent_activations[trialsplit==i]
        thissplit_sep = congruent_separations[trialsplit==i]
        
        distance_per_timestep = []
        # Generate data
        for t in range(n_timesteps):
            theseseps = thissplit_sep[:, t]
            list_of_x = []
            list_of_y = []
            for s in theseseps:
                x, y = generate_data(s, seed=seed*t)
                assert(np.all(y[:int(len(y)/2)]==0) and np.all(y[int(len(y)/2):]==1))
                list_of_x.append(x)
                list_of_y.append(y)
            combx, comby = average_data(list_of_x, list_of_y)
            dist = trainandtest_svm(combx, comby, seed=seed*t)
            distance_per_timestep.append(dist)
            
        activations_tseries_cong.append(thissplit_act.mean(axis=0))
        distances_tseries_cong.append(np.array(distance_per_timestep).reshape(1, -1))
        
    activations_tseries_cong = np.mean(np.vstack(activations_tseries_cong), axis=0)
    distances_tseries_cong = np.mean(np.vstack(distances_tseries_cong), axis=0)
        
    
    # Incongruent
    
    activations_tseries_incong = incongruent_activations.mean(axis=0)
    distances_tseries_incong = []
    
    # Generate data
    #print('Generating incongruent...')
    for t in range(n_timesteps):
        theseseps = incongruent_separations[:, t]
        list_of_x = []
        list_of_y = []
        for s in theseseps:
            x, y = generate_data(s, seed=t)
            list_of_x.append(x)
            list_of_y.append(y)
        combx, comby = average_data(list_of_x, list_of_y)
        dist = trainandtest_svm(combx, comby)
        distances_tseries_incong.append(dist)
        
    distances_tseries_incong = np.array(distances_tseries_incong)
    
    return distances_tseries_cong.mean(), distances_tseries_incong.mean(), \
        pearsonr(distances_tseries_cong, activations_tseries_cong)[0], \
            pearsonr(distances_tseries_incong, activations_tseries_incong)[0]
        
if __name__=="__main__":
    
    congruent_distances = []
    incongruent_distances = []
    congruent_correlations = []
    incongruent_correlations = []
    for i in range(1000):
        dist_cong, dist_incong, corr_cong, corr_incong = generate_trials(seed=i)
        print(f'Distances: {dist_cong:.3f} vs {dist_incong:.3f}')
        print(f'Correlations: {corr_cong:.3f} vs {corr_incong:.3f}')