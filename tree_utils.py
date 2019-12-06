from ete3 import Tree, NodeStyle, TreeStyle
import numpy as np



def depth(a, cur_depth=0):
    if a.is_root():
        return cur_depth
    return depth(a.up, cur_depth+1)


def nup(a, n):
    if n == 0:
        return a
    return nup(a.up, n-1)



def plot_tree(canopy, folder='results', seed=0):

    t = Tree()
    nstyle = NodeStyle()
    nstyle['fgcolor'] = 'black'
    nstyle['size'] = 0
    t.set_style(nstyle)


    r = lambda: np.random.randint(0,255)
    def get_color():
        return '#%02X%02X%02X' % (r(),r(),r())


    log_path = folder + '/log_' + canopy + '_' + str(seed) + '.csv'
    log = np.loadtxt(log_path, delimiter=',', dtype=np.float)
    labels = np.loadtxt('data/rexa/{}/gtClusters.tsv'.format(canopy), delimiter='\t', dtype=np.int)[:,1]

    # print(log[:,:2].astype(int))



    n = int(np.max(log[:,:2])) + 1
    np.random.seed(4)
    colors = [get_color() for i in range(n)]
    nodes = {}
    for i in np.arange(n)[::-1]:
        node = t.add_child(name=str(i))
        nstyle = NodeStyle()
        nstyle['fgcolor'] = colors[labels[int(i)]]
        nstyle['size'] = 5
        node.set_style(nstyle)
        nodes[i] = node

    counter = 0
    for i,j,link in log:
        i, j = int(i), int(j)


        depth_i = depth(nodes[i])
        depth_j = depth(nodes[j])

        # print(depth_i, depth_j)


        par_i = nup(nodes[i], depth_i-1)
        par_j = nup(nodes[j], depth_j-1)

        if depth_i == depth_j:
            l = nodes[i].detach()
            r = nodes[j].detach()

        else:
            par_i = nup(nodes[i], depth_i-1)
            par_j = nup(nodes[j], depth_j-1)

            # print(par_i)
            # print(par_j)

            l = par_i.detach()
            r = par_j.detach()

        new_node = t.add_child()
        nstyle = NodeStyle()
        nstyle['fgcolor'] = 'black'
        nstyle['size'] = 0
        new_node.set_style(nstyle)


        new_node.add_child(l, name=str(i))
        new_node.add_child(r, name=str(j))


    max_depth = 0
    for node in t.get_leaves():
        d = depth(node)
        max_depth = max(max_depth, d)
    max_depth += 1

    for node in t.get_leaves():
        d = depth(node)
        node.dist = max_depth - d


    ts = TreeStyle()
    ts.show_leaf_name = True
    ts.rotation = 90
    # ts.show_branch_length = True
    # ts.show_branch_support = True
    ts.show_scale = False
    t.show(tree_style=ts)

if __name__ == '__main__':
    canopy = 'robinson_h'
    # canopy = 'mcguire_j'
    canopy = 'allen_d'
    canopy = 'blum_a'
    plot_tree(canopy)