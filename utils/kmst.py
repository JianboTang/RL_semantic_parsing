# -*- coding: utf-8 -*-
import numpy as np
from heapq import heappush, heappop

class PriorityQueue:
    def __init__(self):
        self._queue = []

    def put(self, item, priority):
        heappush(self._queue, (-priority, item))

    def get(self):
        return heappop(self._queue)[-1]

    def size(self):
        return len(self._queue)

    def print_queue(self):
        queue_str = ''
        for item in self._queue:
            queue_str = '{0},{1}'.format(queue_str, item[0])
        # print(queue_str)

class MST:
    def __init__(self, n):
        self.parent = [-1] * (n + 1)

    def __hash__(self):
        h = 0
        for i, v in enumerate(self.parent):
            h += (i + 1) * (v + 1)
        return h

    def __eq__(self, other):
        if other != None and isinstance(other, MST):
            if len(self.parent) == len(other.parent):
                for i in xrange(len(self.parent)):
                    if self.parent[i] != other.parent[i]:
                        return False
                return True
        return False

class Edge:
    def __init__(self, src, dst, score):
        self.src = src
        self.dst = dst
        self.score = score

        self.org_src = src
        self.org_dst = dst

        self.__hash_code = hash(src) + hash(dst)
        self.__str = '{0}->{1}'.format(str(src), str(dst))

    def __hash__(self):
        return self.__hash_code

    # def __eq__(self, other):
    #     if other == None:
    #         return False
    #     return self.src == other.src and self.dst == other.dst and self.score == other.score
    def __str__(self):
        return self.__str


class EdmondMST:
    def __init__(self):
        pass

    def _transG(self, G):
        edges = {}
        for s in G:
            if s not in edges:
                edges[s] = {}
            for d in G[s]:
                edges[s][d] = Edge(s, d, G[s][d])
        return edges

    def _getCycle(self, bestInEdge):
        visited = set()
        cycle = set()
        p = {}
        for e in bestInEdge.values():
            p[e.dst] = e.src
        edges = bestInEdge.values()
        bestInEdge = p
        for u in bestInEdge:
            if u not in visited:
                current = u
                current_loop = set()
                current_loop.add(u)
                while current in bestInEdge:
                    current = bestInEdge[current]
                    if current in current_loop:
                        # cycle detected
                        cycle.add((bestInEdge[current], current))
                        start = current
                        while bestInEdge[current] != start:
                            current = bestInEdge[current]
                            cycle.add((bestInEdge[current], current))
                        break
                    current_loop.add(current)
                visited = visited.union(current_loop)
        cycle_edge = set()
        for edge in edges:
            if (edge.src, edge.dst) in cycle:
                cycle_edge.add(edge)
        return cycle, cycle_edge

    def _alt(self, e, path, edges, banned = None):
        children = {}
        #O(n)
        for p in path:
            if p.src not in children:
                children[p.src] = set()
            children[p.src].add(p.dst)
        forb_node = set()
        forb_node.add(e.dst)
        q = list(children[e.src])
        while len(q) > 0:
            subnodes = children[q[0]] if q[0] in children else None
            q.pop(0)
            if subnodes != None:
                q.extend(list(subnodes))
                forb_node = forb_node.union(subnodes)
        maximum = -np.inf
        node = None
        for s in edges:
            if s not in forb_node:
                if e.dst in edges[s]:
                    for alt in edges[s][e.dst]:
                        if alt != e and alt.score > maximum and (banned == None or (alt.org_src, alt.org_dst) not in banned):
                            maximum = alt.score
                            node = alt
        return node


    def _best_score(self, edge, v, other_than = None):
        maximum = -np.inf
        idx = None
        for u in edge:
            if v in edge[u]:
                for e in edge[u][v]:
                    if e.score > maximum and u != v:
                        if other_than != None and e in other_than:
                            continue
                        maximum = e.score
                        idx = e
        return idx#, score[idx][v]

    def tran_score(self, graph, root = 0):
        nr, nc = np.shape(graph)
        G = {}
        for i in xrange(nr):
            if i not in G:
                G[i] = {}
            for j in xrange(nc):
                if j != root and i != j:
                    G[i][j] = graph[i,j]
                if i == root and j != root:
                    G[i][j] -= nr
        return G

    def trans_V(self, G):
        v = set()
        for s in G:
            v.add(s)
        return v

    def tran_E(self, G):
        E = set()
        for s in G:
            for t in G[s]:
                E.add(Edge(s, t, G[s][t]))
        return E

    def tran_edges(self, E):
        edges = {}
        for e in E:
            if e.src not in edges:
                edges[e.src] = {}
            if e.dst not in edges[e.src]:
                edges[e.src][e.dst] = set()
            edges[e.src][e.dst].add(e)
        return edges

    def _prune_edges(self, vertice, edges):
        _s = {}
        for u in vertice:
            if u not in _s:
                _s[u] = {}
            for v in vertice:
                if u in edges and v in edges[u]:
                    _s[u][v] = edges[u][v]
        return _s

    def _init_edges(self,u,v, edges):
        if u not in edges:
            edges[u] = {}
        if v not in edges[u]:
            edges[u][v] = set()

    def get_constraint1best(self, V, E, reqd, banned):
        edges = self.tran_edges(E)
        for (s,d) in banned:
            if s in edges and d in edges[s]:
                E = E - edges[s][d]
                edges[s].pop(d, None)
        for (s,d) in reqd:
            for src in edges:
                if d in edges[src]:
                    if src != s:
                        E = E - edges[src][d]
                        edges[src].pop(d, None)
        t = self.get1best(V, E, edges)
        check_vertice = set()
        for e in t.values():
            check_vertice.add(e.src)
            check_vertice.add(e.dst)
        if len(check_vertice) < len(V):
            return None
        return t

    def get1best(self, V, E, edges = None, root = 0):
        if edges == None:
            edges = self.tran_edges(E)
        bestInEdge = {}
        kickOut = {}
        real = {}
        for v in V - set([root]):
            bestInEdge[v] = self._best_score(edges, v)
            if None == bestInEdge[v]:
                bestInEdge.pop(v, None)
                continue
            Ce, CE = self._getCycle(bestInEdge)
            C = set()
            for (t, u) in Ce:
                C.add(t)
                C.add(u)
            if len(C) > 0:
                vc = max(V) + 1
                _V = V.union(set([vc])) - C
                _E = set()
                for e in E:
                    _e = None
                    if e.src not in C and e.dst not in C:
                        _e = e
                    elif e.src in C and e.dst not in C:
                        _e = Edge(vc, e.dst, e.score)
                    elif e.dst in C and e.src not in C:
                        _e = Edge(e.src, vc, 0)
                        kickOut[_e] = bestInEdge[e.dst]
                        _e.score = e.score - bestInEdge[e.dst].score
                    if _e != None:
                        real[_e] = e
                        _E.add(_e)
                        self._init_edges(_e.src, _e.dst, edges)
                        edges[_e.src][_e.dst].add(_e)
                A = self.get1best(_V, _E, self._prune_edges(_V, edges))
                result = {}
                for u in A:
                    edge = A[u]
                    if edge in real:
                        # result.add(real[edge])
                        result[real[edge].dst] = real[edge]
                if A[vc] in kickOut:
                    CE = CE - set([kickOut[A[vc]]])
                for edge in CE:
                    result[edge.dst] = edge
                return result
        return bestInEdge

    def _uv_max(self, edges, u, v, banned):
        max = -np.inf
        result = None
        if u in edges:
            if v in edges[u]:
                for e in edges[u][v]:
                    if e.score > max and (e.org_src, e.org_dst) not in banned:
                        max = e.score
                        result = e
        return result

    def _prune_none_req(self, edges, reqd):
        req_edge = None
        for e in edges:
            if (e.org_src, e.org_dst) in reqd:
                req_edge = e
                break
        if req_edge:
            edges.clear()
            edges.add(req_edge)

    def find_edge_to_ban(self, A, V, edges, reqd, banned, root = 0):
        path = set()
        for p in A.values():
            path = path.union(edges[p.src][p.dst])
        none_root_edges = set()
        root_edges = set()
        # banned_edges = set()
        for e in path:
            if (e.org_src, e.org_dst) in reqd:
                continue;
            if e.src == root:
                root_edges.add(e)
            else:
                none_root_edges.add(e)

        alt = {}
        diff = {}
        for edge in none_root_edges:
            # best_edge = self._best_score(edges, edge.dst, path)
            best_edge = self._alt(edge, path, edges, banned)
            if best_edge != None:
                #merge nodes
                alt[edge] = best_edge
                diff[edge] = edge.score - best_edge.score

            #merge cycle
            vc = max(V) + 1
            V = (V - set([edge.src, edge.dst])).union(set([vc]))
            _edges = {}
            C = set([edge.src, edge.dst])
            for src in edges:
                for dst in edges[src]:
                    if src in C and dst not in C:
                        self._init_edges(vc, dst, _edges)
                        _edges[vc][dst] = _edges[vc][dst].union(edges[src][dst])
                        # self._prune_none_req(_edges[vc][dst], reqd)
                    if src not in C and dst in C:
                        for e in edges[src][dst]:
                            best_in = self._uv_max(edges, edge.src if edge.src != dst else edge.dst, dst, banned)
                            _e = e
                            _e.score = e.score - best_in.score if best_in != None else np.inf
                            self._init_edges(src, vc, _edges)
                            _edges[src][vc].add(_e)
                            # self._prune_none_req(_edges[src][vc], reqd)
                    if src not in C and dst not in C:
                        self._init_edges(src, dst, _edges)
                        _edges[src][dst] = edges[src][dst]
            edges = _edges

            path.remove(edge)
            for e in path:
                if e.src in C:
                    e.src = vc
                if e.dst in C:
                    e.dst = vc
        for edge in root_edges:
            best_edge = self._alt(edge, path, edges, banned)
                # self._best_score(edges, edge.dst, path)
            if best_edge != None:
                alt[edge] = best_edge
                diff[edge] = edge.score - best_edge.score

        minimum = np.inf
        e_ban = None
        for e in diff:
            if diff[e] < minimum and (e.org_src, e.org_dst) not in reqd and (e.org_src, e.org_dst) not in banned:
                minimum = diff[e]
                e_ban = e
        return e_ban, minimum

    def _has_cycle(self, edges):
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

    def _score_of_path(self, A, G):
        score = 0
        for edge in A.values():
            score += G[edge.src][edge.dst]
        return score

    def _reverse(self, A, n):
        g = MST(n)
        for edge in A.values():
            g.parent[edge.dst] = edge.src
        return g

    def get_kbest(self, G, k):
        reqd = set()
        banned = set()
        q = PriorityQueue()
        A = [None] * k
        E = self.tran_E(G)
        vertice = self.trans_V(G)
        A[0] = self.get1best(self.trans_V(G), E, self.tran_edges(E))
        eban, diff = self.find_edge_to_ban(A[0], self.trans_V(G), self.tran_edges(self.tran_E(G)), reqd, banned)
        score_A = self._score_of_path(A[0], G)
        q.put((score_A - diff, eban, A[0], reqd, banned), score_A - diff)
        record = set()
        record.add(self._reverse(A[0], len(vertice)))
        for j in range(1, k):
            if q.size() == 0:
                break
            (wt, eban, _A, reqd, banned) = q.get()
            # print(wt)
            if wt == -np.inf:
                return A
            _reqd = reqd.union(set([(eban.org_src, eban.org_dst)]))
            _banned = banned.union(set([(eban.org_src, eban.org_dst)]))
            try:
                A[j] = self.get_constraint1best(self.trans_V(G), self.tran_E(G), reqd, _banned)
                _t = self._reverse(A[j], len(vertice))
                if _t in record:
                    j -= 1
                    # continue
                record.add(_t)
                # print('------------wt:{0}'.format(wt))
                # print('reqd: {0}'.format(str(reqd)))
                # print('banned: {0}'.format(str(_banned)))
                # print(self._score_of_path(A[j], G))
                # for edge in A[j].values():
                #     print(str(edge))
                if A[j] != None:
                    eban, diff = self.find_edge_to_ban(_A, self.trans_V(G), self.tran_edges(self.tran_E(G)), _reqd, banned)
                    _A_score = self._score_of_path(_A, G)
                    q.put((_A_score - diff, eban, _A, _reqd, banned), _A_score - diff)
                    eban, diff = self.find_edge_to_ban(_A, self.trans_V(G), self.tran_edges(self.tran_E(G)), reqd, _banned)
                    q.put((_A_score - diff, eban, A[j], reqd, _banned), _A_score - diff)
                else:
                    j -= 1
            except:
                j -= 1
        return A

if __name__ == '__main__':
    mst = EdmondMST()
    scores = np.array([[0, 5, 1, 1], [0, 0, 11, 4], [0, 10, 0, 5], [0, 9, 8, 0]])
    g = mst.tran_score(scores)
    E = mst.tran_E(g)
    A = mst.get1best(mst.trans_V(g), E)
    for edge in A.values():
        print('{0}->{1}'.format(edge.src, edge.dst))


    results = mst.get_kbest(g, 50)
    # for r in results:
    #     if r:
    #         print('--------------')
    #         print(mst._score_of_path(r, g))
    #         for edge in r.values():
    #             print(str(edge))
    # g = mst.tran_score(scores)
    # e_ban, diff = mst.find_edge_to_ban(A, mst.trans_V(g), mst.tran_edges(mst.tran_E(g)), set(), set())
    # print('{0}, diff: {1}'.format(str(e_ban), str(diff)))
    # e_ban, diff = mst.find_edge_to_ban(A, mst.trans_V(g), mst.tran_edges(mst.tran_E(g)), set([(0,1)]), set())
    # print('{0}, diff: {1}'.format(str(e_ban), str(diff)))
    # e_ban, diff = mst.find_edge_to_ban(A, mst.trans_V(g), mst.tran_edges(mst.tran_E(g)), set(), set([(0,3)]))
    # print('{0}, diff: {1}'.format(str(e_ban), str(diff)))
    # e_ban, diff = mst.find_edge_to_ban(A, mst.trans_V(g), mst.tran_edges(mst.tran_E(g)), set([(0,1), (2,3)]), set())
    # print('{0}, diff: {1}'.format(str(e_ban), str(diff)))
    # e_ban, diff = mst.find_edge_to_ban(A, mst.trans_V(g), mst.tran_edges(mst.tran_E(g)), set(), set([(0,3), (0,1)]))
    # print('{0}, diff: {1}'.format(str(e_ban), str(diff)))
    # results = mst.get_kbest(g, 50)

