import sys
import numpy as np

# def _input(filename):
#     prices = {}
#     names = {}
#
#     for line in file(filename).readlines():
#         (name, src, dst, price) = line.rstrip().split()
#         name = int(name.replace('M', ''))
#         src = int(src.replace('C', ''))
#         dst = int(dst.replace('C', ''))
#         price = int(price)
#         t = (src, dst)
#         if t in prices and prices[t] <= price:
#             continue
#         prices[t] = price
#         names[t] = name
#
#     return prices, names


# def _load(arcs, weights):
#     g = {}
#     for (src, dst) in arcs:
#         if src in g:
#             g[src][dst] = weights[(src, dst)]
#         else:
#             g[src] = {dst: weights[(src, dst)]}
#     return g

class Edge(object):
    def __init__(self, parent, child, score):
        self.parent = parent
        self.child = child
        self.score = score

    def __hash__(self):
        return hash(self.parent) + hash(self.child) + hash(self.score)

    def __eq__(self, other):
        return self.parent == other.parent and self.child == other.child and self.score == other.score

    def __cmp__(self, other):
        # if self.score != other.score:
        return cmp(self.score, other.score)
        # if self.parent != other.parent:
        #     return cmp(self.parent, other.parent)
        # if self.child != other.child:
        #     return cmp(self.child, other.child)

def load_scores(scores, root = 0):
    g = {}
    nr, nc = np.shape(scores)
    for src in xrange(nr):
        for dst in xrange(nc):
            if src != dst and dst != root:
                if src in g:
                    g[src][dst] = scores[src, dst]
                else:
                    g[src] = {dst: scores[src, dst]}
    return g

def _reverse(graph):
    r = {}
    for src in graph:
        for (dst, c) in graph[src].items():
            if dst in r:
                r[dst][src] = c
            else:
                r[dst] = {src: c}
    return r

def _has_cycle(edges):
    g = {}
    for (s, d) in edges:
        if d not in g:
            g[d] = s
        else:
            raise Exception('_has_cycle: multi parent exception')
    no_loop_node = set()
    for d in g:
        current = d
        visited = set()
        visited.add(d)
        while current in g:
            if current in no_loop_node:
                break
            current = g[current]
            if current in visited:
                return True
            visited.add(current)
        no_loop_node = no_loop_node.union(visited)
    return False

# def find_edge_to_ban(G, edges, reqd, banned):
#     #naive implementation
#     available_edges = edges - reqd
#     G = G.copy()
#     for (s, d) in banned:
#     #abandon all edge in banned
#         if s in G:
#             G[s].pop(d, None)
#     for (s, d) in reqd:
#     # abandon all (x, d) which x is other than s
#         for o in G:
#             if o != s:
#                 G[o].pop(d, None)
#     org_score = 0
#     for (s, d) in edges:
#         org_score += G[s][d]
#     min_diff = np.inf
#     to_ban = None
#     for edge in available_edges:
#         s = edge[0]
#         d = edge[1]
#         score = G[s].pop(d)
#         h = mst(0, G)
#         new_score = 0
#         for parent in h:
#             for child in h[parent]:
#                 new_score += G[parent][child]
#
#         if org_score - new_score < min_diff:
#             min_diff = org_score
#             to_ban = (s,d)
#         G[s][d] = score
#     return to_ban, min_diff

def find_edge_to_ban(G, edges, reqd, banned):
    # g[src][dst] means the score from parent node src to child node dst
    multi_G = {}
    for s in G:
        for d in G[s]:
            if ((s,d) not in banned or (s,d) in edges) and d != 0:
                if s not in multi_G:
                    multi_G[s] = {d : set([Edge(s, d, G[s][d])])}
                elif d in multi_G[s]:
                    multi_G[s][d].add(Edge(s, d, G[s][d]))
                else:
                    multi_G[s][d] = set([Edge(s, d, G[s][d])])

    path = {}
    for (s,d) in edges:
        if s not in path:
            path[s] = {d: list(multi_G[s][d])[0]}
        else:
            path[s][d] = list(G[s][d])[0]

    reqd_valid = True
    for (s,d) in reqd:
        try:
            if not path[s][d]:
                reqd_valid = False
        except:
            reqd_valid = False
    if not reqd_valid:
        raise Exception("reqd path not in edges")
    reqd = set(list(reqd))
    while len(edges) > 1:
        merged_vertices = set()
        maximum = -np.inf
        e_ban = None
        if len(reqd) > 0:
            for (s, d) in reqd:
                if path[s][d].score > maximum:
                    maximum = path[s][d].score
                    e_ban = (s, d)
            reqd.remove(e_ban)
            edges.remove(e_ban)
        else:
            for (s, d) in edges:
                if path[s][d].score > maximum and (path[s][d].parent, path[s][d].child) not in banned:
                    maximum = path[s][d].score
                    e_ban = (s, d)
            edges.remove(e_ban)
        merged_vertices.add(e_ban[0])
        merged_vertices.add(e_ban[1])

        alt_min = np.inf
        for s in multi_G:
            if s != e_ban[1] and s != e_ban[0]:
                edges.add((s, e_ban[1]))
                if not _has_cycle(edges) and e_ban[1] in multi_G[s]:
                    #calculate diff
                    diff = path[e_ban[0]][e_ban[1]].score - max(multi_G[s][e_ban[1]]).score
                    if diff < alt_min:
                        alt_min = diff
                edges.remove((s, e_ban[1]))
        #merge_cycle
        new_node = max(multi_G.keys()) + 1
        path[new_node] = {}

        for s in path:
            path[s][new_node] = None
            if s not in merged_vertices:
                for d in path[s]:
                    if d == e_ban[0]:
                        path[s][new_node] = path[s][d]
            elif s in merged_vertices:
                for d in path[s]:
                    if d != new_node and d not in merged_vertices:
                        path[new_node][d] = path[s][d]

        multi_G[new_node] = {}
        for s in multi_G:
            if s != new_node:
                multi_G[s][new_node] = set()
            for d in multi_G[s]:
                if s not in merged_vertices and d in merged_vertices and s != new_node:
                    for edge in list(multi_G[s][d]):
                        edge.score -= max(multi_G[e_ban[0] if e_ban[0] != d else e_ban[1]][d]).score
                        multi_G[s][new_node].add(edge)
                if s in merged_vertices and d not in merged_vertices and d != new_node:
                    for edge in list(multi_G[s][d]):
                        if d not in multi_G[new_node]:
                            multi_G[new_node][d] = set()
                        multi_G[new_node][d].add(edge)
                        # multi_G[s][d].add(Edge(s, new_edge, edge.score - max(multi_G[e_ban[0] if e_ban[0] != d else e_ban[1]][d]).score))

        #remove node in multi_G
        multi_G.pop(e_ban[0], 0)
        multi_G.pop(e_ban[1], 0)
        for s in multi_G:
            multi_G[s].pop(e_ban[0], 0)
            multi_G[s].pop(e_ban[1], 0)

        path.pop(e_ban[0], 0)
        path.pop(e_ban[1], 0)
        for s in path:
            path[s].pop(e_ban[0], 0)
            path[s].pop(e_ban[1], 0)

        new_edge = set()
        for (s, d) in edges:
            if d in merged_vertices and s in merged_vertices:
                pass
            elif d in merged_vertices:
                new_edge.add((s, new_node))
            elif s in merged_vertices:
                new_edge.add((new_node, d))
            else:
                new_edge.add((s,d))
        edges = new_edge
    to_ban = list(edges)[0]
    p = path[to_ban[0]][to_ban[1]]
    minimum = np.inf
    for s in multi_G:
        for d in multi_G[s]:
            for edge in multi_G[s][d]:
                if not edge.__eq__(p):
                    if p.score - edge.score < minimum:
                        minimum = p.score - edge.score
    # for edge in multi_G[to_ban[0]][to_ban[1]]:
    #     if not edge.__eq__(p):
    #         if p.score - edge.score < minimum:
    #             minimum = p.score - edge.score

    return (p.parent, p.child), minimum


def _mergeCycles(cycle,G,RG,g,rg):
    allInEdges = []
    minInternal = None
    minInternalWeight = sys.maxint

    # find minimal internal edge weight
    for n in cycle:
        for e in RG[n]:
            if e in cycle:
                if minInternal is None or RG[n][e] < minInternalWeight:
                    minInternal = (n,e)
                    minInternalWeight = RG[n][e]
                    continue
            else:
                allInEdges.append((n,e))

    # find the incoming edge with minimum modified cost
    minExternal = None
    minModifiedWeight = 0
    for s,t in allInEdges:
        u,v = rg[s].popitem()
        rg[s][u] = v
        w = RG[s][t] - (v - minInternalWeight)
        if minExternal is None or minModifiedWeight > w:
            minExternal = (s,t)
            minModifiedWeight = w

    u,w = rg[minExternal[0]].popitem()
    rem = (minExternal[0],u)
    rg[minExternal[0]].clear()
    if minExternal[1] in rg:
        rg[minExternal[1]][minExternal[0]] = w
    else:
        rg[minExternal[1]] = { minExternal[0] : w }
    if rem[1] in g:
        if rem[0] in g[rem[1]]:
            del g[rem[1]][rem[0]]
    if minExternal[1] in g:
        g[minExternal[1]][minExternal[0]] = w
    else:
        g[minExternal[1]] = { minExternal[0] : w }


# --------------------------------------------------------------------------------- #


def _getCycle(bestInEdge):
    visited = set()
    for u in bestInEdge:
        if u not in visited:
            current = u
            current_loop = set()
            current_loop.add(u)
            while current in bestInEdge:
                current = bestInEdge[u]
                if current in current_loop:
                    # cycle detected
                    cycle = set()
                    cycle.add(current)
                    start = current
                    while bestInEdge[current] != start:
                        current = bestInEdge[current]
                        cycle.add(current)
                    break
            visited = visited.union(current)

def mst(V, E, root):
    V = V - root
    bestInEdge = {}
    for v in V:
        maximum = -np.inf
        idx = -1
        for s in E:
            if v in E[s]:
                if E[s][v] > maximum:
                    maximum = E[s][v]
                    idx = s
        bestInEdge[v] = idx
    cycles = _getCycle(bestInEdge)

# --------------------------------------------------------------------------------- #

if __name__ == "__main__":
    # try:
    #     filename = sys.argv[1]
    #     root = sys.argv[2]
    # except IndexError:
    #     sys.stderr.write('no input and/or root node specified\n')
    #     sys.stderr.write('usage: python edmonds.py <file> <root>\n')
    #     sys.exit(1)

    # test_score = np.transpose(np.array([[-1,-1,-1,-1], [0,0,1,0], [1,0,0,0], [0,0,1,0]]))
    # g = load_scores(test_score)
    # h = mst(0, g)
    # for s in h:
    #     for t in h[s]:
    #         print "%d-%d" % (s, t)

    scores = np.array([[0, 5, 1, 1], [0, 0, 11, 4], [0, 10, 0, 5], [0, 9, 8, 0]])
    g = load_scores(scores)
    # h = mst(0, g)
    # print(h)
    # g = load_scores(scores)
    # print(find_edge_to_ban(g, set([(0,1), (1,2), (2,3)]), set(), set()))
    #print(find_edge_to_ban(g, set([(0,1), (1,2), (2,3)]), set([(0,1)]), set())) #((2,3), 1)
    print(find_edge_to_ban(g, set([(0,1), (1,2), (2,3)]), set(), set([(1,2)]))) #(


    # a = set()
    # a.add((1,2))
    # a.add((3,4))
    # a.add((1,2))
    # a.remove((1,2))
    # print(a)
    # prices, names = _input(filename)
    # g = _load(prices, prices)
    # h = mst(int(root), g)
    # for s in h:
    #     for t in h[s]:
    #         print "%d-%d" % (s, t)