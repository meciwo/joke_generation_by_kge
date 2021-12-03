import networkx as nx
import configparser
import matplotlib.pyplot as plt
import pandas as pd

config = configparser.ConfigParser()
config.read("config.ini")
kg_path = "data/knowlegegraph_short_words.csv"
train_path = config["Paths"]["TrainDataPath"]
ent_vocab_path = config["Paths"]["EntVocabPath"]
rel_vocab_path = config["Paths"]["RelVocabPath"]


def load_network():
    G = nx.Graph()
    ke = pd.read_csv(kg_path)

    """
    with open(train_path, "r") as f:
        for i, line in enumerate(f):
            head, relation, tail = line.strip().split(" ")
            G.add_nodes_from([head, tail])
            G.add_edge(head, tail)
    """

    for head, relation, tail in ke[["head", "relation", "tail"]].values:
        # G.add_nodes_from([head, tail])
        G.add_edge(head, tail, relation=relation)
    return G


def find_nodes_related_edge(G, name_of_relation, shortest_nodes):
    nx.get_edge_attributes(G, "relation")
    _map = set()
    for k, v in nx.get_edge_attributes(G, "relation").items():
        if v == name_of_relation:
            if k[0] in shortest_nodes or k[1] in shortest_nodes:
                _map.add(k[0])
                _map.add(k[1])
    return list(_map)


def visualize(head, relation, tail):
    G = load_network()
    shortest_nodes = nx.shortest_path(G, head, tail)
    linked_nodes = find_nodes_related_edge(G, relation, shortest_nodes)
    sub = G.subgraph(linked_nodes + shortest_nodes)

    nx.nx_agraph.view_pygraphviz(sub, prog="fdp")  # pygraphvizが必要
    plt.show()

    shortest_sub = G.subgraph(shortest_nodes)
    pos = nx.spring_layout(shortest_sub)
    plt.figure()
    nx.draw(
        shortest_sub,
        pos,
        edge_color="black",
        width=1,
        linewidths=1,
        node_size=500,
        node_color="pink",
        alpha=0.9,
    )
    nx.draw_networkx_edge_labels(
        shortest_sub,
        pos=pos,
        edge_labels=nx.get_edge_attributes(shortest_sub, "relation"),
        font_color="black",
        font_size=15,
    )
