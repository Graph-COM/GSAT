# From Discovering Invariant Rationales for Graph Neural Networks

from .BA3_loc import *
from pathlib import Path
import random
from tqdm import tqdm


def gen_dataset(global_b, data_path):
    n_node = 0
    n_edge = 0
    for _ in range(1000):
        # small:
        width_basis=np.random.choice(range(3,4))     # tree    #Node 32.55 #Edge 35.04
        # width_basis=np.random.choice(range(8,12))  # ladder  #Node 24.076 #Edge 34.603
        # width_basis=np.random.choice(range(15,20)) # wheel   #Node 21.954 #Edge 40.264
        # large:
        # width_basis=np.random.choice(range(3,6))   # tree    #Node 111.562 #Edge 117.77
        # width_basis=np.random.choice(range(30,50)) # ladder  #Node 83.744 #Edge 128.786
        # width_basis=np.random.choice(range(60,80)) # wheel   #Node 83.744 #Edge 128.786
        G, role_id, name = get_crane(basis_type="tree", nb_shapes=1,
                                            width_basis=width_basis,
                                            feature_generator=None, m=3, draw=False)
        role_id = np.array(role_id)
        edge_index = np.array(G.edges, dtype=int).T
        row, col = edge_index
        ground_truth = find_gd(edge_index, role_id)

    #     pos = nx.spring_layout(G)
    #     nx.draw_networkx_nodes(G, pos=pos, nodelist=range(len(G.nodes())), node_size=150,
    #                            node_color=role_id, cmap='bwr',
    #                            linewidths=.1, edgecolors='k')

    #     nx.draw_networkx_labels(G, pos,
    #                             labels={i: str(role_id[i]) for i in range(len(G.nodes))},
    #                             font_size=10,
    #                             font_weight='bold', font_color='k'
    #                             )
    #     nx.draw_networkx_edges(G, pos=pos, edgelist=G.edges(), edge_color='black')
    #     plt.show()

        n_node += len(role_id)
        n_edge += edge_index.shape[1]
    print("#Node", n_node/1000, "#Edge", n_edge/1000)

    # Training Dataset
    edge_index_list = []
    label_list = []
    ground_truth_list = []
    role_id_list = []
    pos_list = []

    bias = float(global_b)
    e_mean = []
    n_mean = []
    for _ in tqdm(range(3000)):
        base_num = np.random.choice([1,2,3], p=[bias,(1-bias)/2,(1-bias)/2])

        if base_num == 1:
            base = 'tree'
            width_basis=np.random.choice(range(3,4))
        if base_num == 2:
            base = 'ladder'
            width_basis=np.random.choice(range(8,12))
        if base_num == 3:
            base = 'wheel'
            width_basis=np.random.choice(range(15,20))

        G, role_id, name = get_cycle(basis_type=base, nb_shapes=1,
                                        width_basis=width_basis, feature_generator=None, m=3, draw=False)
        role_id = np.array(role_id)
        edge_index = np.array(G.edges, dtype=int).T
        row, col = edge_index
        e_mean.append(len(G.edges))
        n_mean.append(len(G.nodes))
        edge_index_list.append(edge_index)
        label_list.append(0)
        ground_truth = find_gd(edge_index, role_id)
        ground_truth_list.append(ground_truth)
        role_id_list.append(role_id)
        pos = nx.spring_layout(G)
        pos_list.append(pos)
    print(np.mean(n_mean), np.mean(e_mean))
    print(len(ground_truth_list))

    e_mean = []
    n_mean = []
    for _ in tqdm(range(3000)):
        base_num = np.random.choice([1,2,3], p=[(1-bias)/2,bias,(1-bias)/2])

        if base_num == 1:
            base = 'tree'
            width_basis=np.random.choice(range(3))
        if base_num == 2:
            base = 'ladder'
            width_basis=np.random.choice(range(8,12))
        if base_num == 3:
            base = 'wheel'
            width_basis=np.random.choice(range(15,20))

        G, role_id, name = get_house(basis_type=base, nb_shapes=1,
                                        width_basis=width_basis, feature_generator=None, m=3, draw=False)
        role_id = np.array(role_id)
        edge_index = np.array(G.edges, dtype=int).T
        row, col = edge_index
        e_mean.append(len(G.edges))
        n_mean.append(len(G.nodes))
        edge_index_list.append(edge_index)
        label_list.append(1)
        ground_truth = find_gd(edge_index, role_id)
        ground_truth_list.append(ground_truth)
        role_id_list.append(role_id)
        pos = nx.spring_layout(G)
        pos_list.append(pos)
    print(np.mean(n_mean), np.mean(e_mean))
    print(len(ground_truth_list))

    e_mean = []
    n_mean = []
    for _ in tqdm(range(3000)):
        base_num = np.random.choice([1,2,3], p=[(1-bias)/2,(1-bias)/2,bias])

        if base_num == 1:
            base = 'tree'
            width_basis=np.random.choice(range(3))
        if base_num == 2:
            base = 'ladder'
            width_basis=np.random.choice(range(8,12))
        if base_num == 3:
            base = 'wheel'
            width_basis=np.random.choice(range(15,20))

        G, role_id, name = get_crane(basis_type=base, nb_shapes=1,
                                        width_basis=width_basis, feature_generator=None, m=3, draw=False)
        role_id = np.array(role_id)
        edge_index = np.array(G.edges, dtype=int).T
        row, col = edge_index
        e_mean.append(len(G.edges))
        n_mean.append(len(G.nodes))
        edge_index_list.append(edge_index)
        label_list.append(2)
        ground_truth = find_gd(edge_index, role_id)
        ground_truth_list.append(ground_truth)
        role_id_list.append(role_id)
        pos = nx.spring_layout(G)
        pos_list.append(pos)
    print(np.mean(n_mean), np.mean(e_mean))
    print(len(ground_truth_list))

    np.save(data_path / 'train.npy', (edge_index_list, label_list, ground_truth_list, role_id_list, pos_list))


    # Validation Dataset
    edge_index_list = []
    label_list = []
    ground_truth_list = []
    role_id_list = []
    pos_list = []

    bias = 1.0/3
    e_mean = []
    n_mean = []
    for _ in tqdm(range(1000)):
        base_num = np.random.choice([1,2,3], p=[bias,(1-bias)/2,(1-bias)/2])

        if base_num == 1:
            base = 'tree'
            width_basis=np.random.choice(range(3,4))
        if base_num == 2:
            base = 'ladder'
            width_basis=np.random.choice(range(8,12))
        if base_num == 3:
            base = 'wheel'
            width_basis=np.random.choice(range(15,20))

        G, role_id, name = get_cycle(basis_type=base, nb_shapes=1,
                                        width_basis=width_basis, feature_generator=None, m=3, draw=False)
        role_id = np.array(role_id)
        edge_index = np.array(G.edges, dtype=int).T
        row, col = edge_index
        e_mean.append(len(G.edges))
        n_mean.append(len(G.nodes))
        edge_index_list.append(edge_index)
        label_list.append(0)
        ground_truth = find_gd(edge_index, role_id)
        ground_truth_list.append(ground_truth)
        role_id_list.append(role_id)
        pos = nx.spring_layout(G)
        pos_list.append(pos)
    print(np.mean(n_mean), np.mean(e_mean))
    print(len(ground_truth_list))

    e_mean = []
    n_mean = []
    for _ in tqdm(range(1000)):
        base_num = np.random.choice([1,2,3], p=[(1-bias)/2,bias,(1-bias)/2])

        if base_num == 1:
            base = 'tree'
            width_basis=np.random.choice(range(3,4))
        if base_num == 2:
            base = 'ladder'
            width_basis=np.random.choice(range(8,12))
        if base_num == 3:
            base = 'wheel'
            width_basis=np.random.choice(range(15,20))

        G, role_id, name = get_house(basis_type=base, nb_shapes=1,
                                        width_basis=width_basis, feature_generator=None, m=3, draw=False)
        role_id = np.array(role_id)
        edge_index = np.array(G.edges, dtype=int).T
        row, col = edge_index
        e_mean.append(len(G.edges))
        n_mean.append(len(G.nodes))
        edge_index_list.append(edge_index)
        label_list.append(1)
        ground_truth = find_gd(edge_index, role_id)
        ground_truth_list.append(ground_truth)
        role_id_list.append(role_id)
        pos = nx.spring_layout(G)
        pos_list.append(pos)
    print(np.mean(n_mean), np.mean(e_mean))
    print(len(ground_truth_list))

    e_mean = []
    n_mean = []
    for _ in tqdm(range(1000)):
        base_num = np.random.choice([1,2,3], p=[(1-bias)/2,(1-bias)/2,bias])

        if base_num == 1:
            base = 'tree'
            width_basis=np.random.choice(range(3,4))
        if base_num == 2:
            base = 'ladder'
            width_basis=np.random.choice(range(8,12))
        if base_num == 3:
            base = 'wheel'
            width_basis=np.random.choice(range(15,20))

        G, role_id, name = get_crane(basis_type=base, nb_shapes=1,
                                        width_basis=width_basis, feature_generator=None, m=3, draw=False)
        role_id = np.array(role_id)
        edge_index = np.array(G.edges, dtype=int).T
        row, col = edge_index
        e_mean.append(len(G.edges))
        n_mean.append(len(G.nodes))
        edge_index_list.append(edge_index)
        label_list.append(2)
        ground_truth = find_gd(edge_index, role_id)
        ground_truth_list.append(ground_truth)
        role_id_list.append(role_id)
        pos = nx.spring_layout(G)
        pos_list.append(pos)
    print(np.mean(n_mean), np.mean(e_mean))
    print(len(ground_truth_list))
    np.save(data_path / 'val.npy', (edge_index_list, label_list, ground_truth_list, role_id_list, pos_list))

    # Test Dataset

    edge_index_list = []
    label_list = []
    ground_truth_list = []
    role_id_list = []
    pos_list = []

    e_mean = []
    n_mean = []
    for _ in tqdm(range(2000)):
        base_num = np.random.choice([1,2,3])

        if base_num == 1:
            base = 'tree'
            width_basis=np.random.choice(range(3,6))
        if base_num == 2:
            base = 'ladder'
            width_basis=np.random.choice(range(30,50))
        if base_num == 3:
            base = 'wheel'
            width_basis=np.random.choice(range(60,80))

        G, role_id, name = get_cycle(basis_type=base, nb_shapes=1,
                                        width_basis=width_basis, feature_generator=None, m=3, draw=False)
        role_id = np.array(role_id)
        edge_index = np.array(G.edges, dtype=int).T
        row, col = edge_index
        e_mean.append(len(G.edges))
        n_mean.append(len(G.nodes))
        edge_index_list.append(edge_index)
        label_list.append(0)
        ground_truth = find_gd(edge_index, role_id)
        ground_truth_list.append(ground_truth)
        role_id_list.append(role_id)
        pos = nx.spring_layout(G)
        pos_list.append(pos)
    print(np.mean(n_mean), np.mean(e_mean))
    print(len(ground_truth_list))

    e_mean = []
    n_mean = []
    for _ in tqdm(range(2000)):
        base_num = np.random.choice([1,2,3])

        if base_num == 1:
            base = 'tree'
            width_basis=np.random.choice(range(3,6))
        if base_num == 2:
            base = 'ladder'
            width_basis=np.random.choice(range(30,50))
        if base_num == 3:
            base = 'wheel'
            width_basis=np.random.choice(range(60,80))

        G, role_id, name = get_house(basis_type=base, nb_shapes=1,
                                        width_basis=width_basis, feature_generator=None, m=3, draw=False)
        role_id = np.array(role_id)
        edge_index = np.array(G.edges, dtype=int).T
        row, col = edge_index
        e_mean.append(len(G.edges))
        n_mean.append(len(G.nodes))
        edge_index_list.append(edge_index)
        label_list.append(1)
        ground_truth = find_gd(edge_index, role_id)
        ground_truth_list.append(ground_truth)
        role_id_list.append(role_id)
        pos = nx.spring_layout(G)
        pos_list.append(pos)
    print(np.mean(n_mean), np.mean(e_mean))
    print(len(ground_truth_list))

    e_mean = []
    n_mean = []
    for _ in tqdm(range(2000)):
        base_num = np.random.choice([1,2,3])

        if base_num == 1:
            base = 'tree'
            width_basis=np.random.choice(range(3,6))
        if base_num == 2:
            base = 'ladder'
            width_basis=np.random.choice(range(30,50))
        if base_num == 3:
            base = 'wheel'
            width_basis=np.random.choice(range(60,80))

        G, role_id, name = get_crane(basis_type=base, nb_shapes=1,
                                        width_basis=width_basis, feature_generator=None, m=3, draw=False)
        role_id = np.array(role_id)
        edge_index = np.array(G.edges, dtype=int).T
        row, col = edge_index
        e_mean.append(len(G.edges))
        n_mean.append(len(G.nodes))
        edge_index_list.append(edge_index)
        label_list.append(2)
        ground_truth = find_gd(edge_index, role_id)
        ground_truth_list.append(ground_truth)
        role_id_list.append(role_id)
        pos = nx.spring_layout(G)
        pos_list.append(pos)
    print(np.mean(n_mean), np.mean(e_mean))
    print(len(ground_truth_list))
    np.save(data_path / 'test.npy', (edge_index_list, label_list, ground_truth_list, role_id_list, pos_list))


def get_house(basis_type, nb_shapes=80, width_basis=8, feature_generator=None, m=3, draw=True):
    """ Synthetic Graph #5:

    Start with a tree and attach grid-shaped subgraphs.

    Args:
        nb_shapes         :  The number of shapes (here 'houses') that should be added to the base graph.
        width_basis       :  The width of the basis graph (here a random 'grid').
        feature_generator :  A `FeatureGenerator` for node features. If `None`, add constant features to nodes.
        m                 :  The tree depth.

    Returns:
        G                 :  A networkx graph
        role_id           :  Role ID for each node in synthetic graph
        name              :  A graph identifier
    """
    list_shapes = [["house"]] * nb_shapes

    if draw:
        plt.figure(figsize=figsize)

    G, role_id, _ = synthetic_structsim.build_graph(
        width_basis, basis_type, list_shapes, start=0, rdm_basis_plugins=True
    )
    G = perturb([G], 0.05, id=role_id)[0]

    if feature_generator is None:
        feature_generator = featgen.ConstFeatureGen(1)
    feature_generator.gen_node_features(G)

    name = basis_type + "_" + str(width_basis) + "_" + str(nb_shapes)

    return G, role_id, name


def get_cycle(basis_type, nb_shapes=80, width_basis=8, feature_generator=None, m=3, draw=True):
    """ Synthetic Graph #5:

    Start with a tree and attach grid-shaped subgraphs.

    Args:
        nb_shapes         :  The number of shapes (here 'houses') that should be added to the base graph.
        width_basis       :  The width of the basis graph (here a random 'grid').
        feature_generator :  A `FeatureGenerator` for node features. If `None`, add constant features to nodes.
        m                 :  The tree depth.

    Returns:
        G                 :  A networkx graph
        role_id           :  Role ID for each node in synthetic graph
        name              :  A graph identifier
    """
    list_shapes = [["dircycle"]] * nb_shapes

    if draw:
        plt.figure(figsize=figsize)

    G, role_id, _ = synthetic_structsim.build_graph(
        width_basis, basis_type, list_shapes, start=0, rdm_basis_plugins=True
    )
    G = perturb([G], 0.05, id=role_id)[0]

    if feature_generator is None:
        feature_generator = featgen.ConstFeatureGen(1)
    feature_generator.gen_node_features(G)

    name = basis_type + "_" + str(width_basis) + "_" + str(nb_shapes)

    return G, role_id, name


def get_crane(basis_type, nb_shapes=80, width_basis=8, feature_generator=None, m=3, draw=True):
    """ Synthetic Graph #5:

    Start with a tree and attach grid-shaped subgraphs.

    Args:
        nb_shapes         :  The number of shapes (here 'houses') that should be added to the base graph.
        width_basis       :  The width of the basis graph (here a random 'grid').
        feature_generator :  A `FeatureGenerator` for node features. If `None`, add constant features to nodes.
        m                 :  The tree depth.

    Returns:
        G                 :  A networkx graph
        role_id           :  Role ID for each node in synthetic graph
        name              :  A graph identifier
    """
    list_shapes = [["crane"]] * nb_shapes

    if draw:
        plt.figure(figsize=figsize)

    G, role_id, _ = synthetic_structsim.build_graph(
        width_basis, basis_type, list_shapes, start=0, rdm_basis_plugins=True
    )
    G = perturb([G], 0.05, id=role_id)[0]

    if feature_generator is None:
        feature_generator = featgen.ConstFeatureGen(1)
    feature_generator.gen_node_features(G)

    name = basis_type + "_" + str(width_basis) + "_" + str(nb_shapes)

    return G, role_id, name
