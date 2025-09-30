import argparse
import os
import pickle
from typing import Dict, List


def load_sessions(dataset: str) -> List[List[int]]:
    path = os.path.join('datasets', dataset, 'all_train_seq.txt')
    if not os.path.exists(path):
        raise FileNotFoundError(f'Missing all_train_seq.txt for dataset "{dataset}" at {path}')
    with open(path, 'rb') as f:
        return pickle.load(f)


def infer_num_nodes(sessions: List[List[int]]) -> int:
    max_item_id = 0
    for session in sessions:
        for item in session:
            if item > max_item_id:
                max_item_id = item
    if max_item_id <= 0:
        raise ValueError('Could not infer a positive item id from the sessions.')
    # IDs are 1-based; allocate one extra slot so we can index max_id directly.
    return max_item_id + 1


def build_global_graph(sessions: List[List[int]], num_nodes: int, sample_num: int):
    adj_dict: List[Dict[int, int]] = [dict() for _ in range(num_nodes)]

    for session in sessions:
        length = len(session)
        for hop in (1, 2, 3):
            if length <= hop:
                break
            for idx in range(length - hop):
                src = session[idx]
                dst = session[idx + hop]
                if src <= 0 or dst <= 0:
                    continue
                adj_dict[src].setdefault(dst, 0)
                adj_dict[src][dst] += 1
                adj_dict[dst].setdefault(src, 0)
                adj_dict[dst][src] += 1

    adj: List[List[int]] = [[] for _ in range(num_nodes)]
    weight: List[List[int]] = [[] for _ in range(num_nodes)]

    for node in range(num_nodes):
        neighbors = adj_dict[node]
        if not neighbors:
            continue
        # Sort by descending co-occurrence frequency.
        sorted_neighbors = sorted(neighbors.items(), key=lambda x: x[1], reverse=True)
        # Keep only the top-k entries.
        limited = sorted_neighbors[:sample_num]
        adj[node] = [item for item, _ in limited]
        weight[node] = [freq for _, freq in limited]

    return adj, weight


def save_graph(dataset: str, sample_num: int, adj, weight):
    base = os.path.join('datasets', dataset)
    with open(os.path.join(base, f'adj_{sample_num}.pkl'), 'wb') as f:
        pickle.dump(adj, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(base, f'num_{sample_num}.pkl'), 'wb') as f:
        pickle.dump(weight, f, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='diginetica', help='Dataset name under datasets/')
    parser.add_argument('--sample_num', type=int, default=12)
    opt = parser.parse_args()

    sessions = load_sessions(opt.dataset)
    num_nodes = infer_num_nodes(sessions)
    print(f'Building global graph for {opt.dataset}: {len(sessions)} sessions, {num_nodes - 1} items (1-based IDs).')

    adj, weight = build_global_graph(sessions, num_nodes, opt.sample_num)
    save_graph(opt.dataset, opt.sample_num, adj, weight)
    print(f'Saved adjacency to adj_{opt.sample_num}.pkl and weights to num_{opt.sample_num}.pkl.')


if __name__ == '__main__':
    main()
