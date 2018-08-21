""" nn_idx_from_node = list(self.edge_graph.get_nn(node))
        if len(nn_idx_from_node) < 2:
            node_2 = nn_idx_from_node[0]
            edge_to_merge = (node, node_2)
            score_edge = self.edge_graph[(node, node_2)].LCB()
        else: # # # ==> multi-merger -> merge an hyper-edge
            print("Multi-merger")
            edge_score_from_node = []
            edge_error_score_from_node = []
            for nn in nn_idx_from_node:
                edge_score_from_node.append(self.edge_graph[(node ,nn)].score)
                edge_error_score_from_node.append(self.edge_graph[(node ,nn)].score_error)

            asort = np.argsort(edge_score_from_node)
            worst = asort[0]
            min_score = edge_score_from_node[worst]
            error_min_score = edge_error_score_from_node[worst]
            edge_to_merge = [node]

            for i in range(1, len(asort)):
                gap = (edge_score_from_node[asort[i]] - min_score)# - (edge_error_score_from_node[asort[i]]+error_min_score)
                if gap > 0: # threshold here !
                    edge_to_merge.append(nn_idx_from_node[asort[i]])
                    break
                else:
                    edge_to_merge.append(nn_idx_from_node[asort[i]])
            score_edge = -1  # undefined
        else:
            # if gap smaller than a certain threshold, ==> Flatten structure, multi-merger
            nn_yu = list(self.edge_graph.get_nn(node))

            edge_ij = []
            for nn in nn_yu:
                    edge_ij.append(self.edge_graph[(node,nn)].LCB())
            asort_minscore_edge = np.argsort(edge_ij)

            edge_to_merge = (node, nn_yu[asort_minscore_edge[0]])

            score_edge = edge_ij[asort_minscore_edge[0]] """