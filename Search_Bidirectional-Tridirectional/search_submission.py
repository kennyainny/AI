# coding=utf-8
"""
This file is your main submission that will be graded against. Only copy-paste
code on the relevant classes included here. Do not add any classes or functions
to this file that are not part of the classes that we want.
"""

from __future__ import division

import Queue
import heapq
import os
import pickle
import math


class PriorityQueue(object):
    """
    A queue structure where each element is served in order of priority.

    Elements in the queue are popped based on the priority with higher priority
    elements being served before lower priority elements.  If two elements have
    the same priority, they will be served in the order they were added to the
    queue.

    Traditionally priority queues are implemented with heaps, but there are any
    number of implementation options.

    (Hint: take a look at the module heapq)

    Attributes:
        queue (list): Nodes added to the priority queue.
        current (int): The index of the current node in the queue.
    """

    def __init__(self):
        """Initialize a new Priority Queue."""

        self.queue = []

    def pop(self):
        """
        Pop top priority node from queue.

        Returns:
            The node with the highest priority.
        """
        if len(self.queue) > 0: #check for heap length first
            return heapq.heappop(self.queue)

        #heappop(h)

        # TODO: finish this function!
        raise NotImplementedError

    def remove(self, node_id):
        """
        Remove a node from the queue.

        This is a hint, you might require this in ucs,
        however, if you choose not to use it, you are free to
        define your own method and not use it.

        Args:
            node_id (int): Index of node in queue.
        """
        self.queue.remove(self.queue[node_id])

        # raise NotImplementedError

    def __iter__(self):
        """Queue iterator."""

        return iter(sorted(self.queue))

    def __str__(self):
        """Priority Queue to string."""

        return 'PQ:%s' % self.queue

    def append(self, node):
        """
        Append a node to the queue.

        Args:
            node: Comparable Object to be added to the priority queue.
        """
        heapq.heappush(self.queue, (node[0], node))
        # print self.queue

        # TODO: finish this function!
        # raise NotImplementedError

    def contains2(self, node_state):
        """

        """
        # node(cost,state,parent,action or path)
        bool_val = any(node_state == q[1][1] for q in self.queue) #list comprehension
        return bool_val

    def getCost(self, node_state):
        """

        """
        # exist_node = [item for item in self.queue if item[1][1] == node_state]
        exist_node = next(item for item in self.queue if item[1][1] == node_state)
        return exist_node[0] #[0]

    def getNode(self, node_state):
        """

        """
        exist_node = next(item for item in self.queue if item[1][1] == node_state)
        return exist_node

    def remove2(self, node_name):
        """

        """
        #check if element exist in the list

        # self.queue.remove(self.queue[node_id])

        existing_node =self.getNode(node_name)
        self.queue.remove(existing_node)

        heapq.heapify(self.queue)
        # raise NotImplementedError


    def __contains__(self, key):
        """
        Containment Check operator for 'in'

        Args:
            key: The key to check for in the queue.

        Returns:
            True if key is found in queue, False otherwise.
        """

        return key in [n for _, n in self.queue]

    def __eq__(self, other):
        """
        Compare this Priority Queue with another Priority Queue.

        Args:
            other (PriorityQueue): Priority Queue to compare against.

        Returns:
            True if the two priority queues are equivalent.
        """

        return self == other

    def size(self):
        """
        Get the current size of the queue.

        Returns:
            Integer of number of items in queue.
        """

        return len(self.queue)

    def clear(self):
        """Reset queue to empty (no nodes)."""

        self.queue = []

    def top(self):
        """
        Get the top item in the queue.

        Returns:
            The first item stored in teh queue.
        """

        return self.queue[0]


def breadth_first_search(graph, start, goal):
    """
    Warm-up exercise: Implement breadth-first-search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    explored_list = []  # set()

    if start == goal:
        return explored_list
    else:
        frontierQueue = Queue.deque()
        frontierQueue.append(start)

        # for temp_node in frontier_list:
        #     frontierQueue.put(temp_node)

        while len(frontierQueue) > 0:
            bfs_path = frontierQueue.popleft()

            child_node = bfs_path[-1]

            if (child_node not in explored_list): #and (child_node not in frontierQueue):
                neighbor_nodes = graph[child_node]

                for child_neighbor in neighbor_nodes:
                    temp_path = list(bfs_path)
                    temp_path.append(child_neighbor)
                    frontierQueue.append(temp_path)

                    if child_neighbor == goal:
                        return temp_path

                explored_list.append(child_node)

        return []

        # TODO: finish this function!
    # raise NotImplementedError


def uniform_cost_search(graph, start, goal):
    """
    Warm-up exercise: Implement uniform_cost_search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    if start == goal:
        return []
    else:
        path_cost = 0
        explored_list = set()
        frontierQueue = PriorityQueue()
        frontierQueue.append((path_cost,start,start,[start])) #node(cost,state,parent,action or path)

        while frontierQueue.size() > 0:

            p_cost,node_state,p_node,n_path = frontierQueue.pop()[1] #node(cost,state,parent,action or path)
            # node_state = state

            if node_state == goal:
                return list(n_path) #list element of the dictionary

            explored_list.add(node_state)

            neighbor_nodes = graph[node_state]  # get the neighbors
            for neighbor in neighbor_nodes:
                cost = graph[node_state][neighbor]['weight']+p_cost
                node_path = n_path[:]
                node_path.append(neighbor)
                child_node = (cost, neighbor, node_state, node_path) #node(cost,state,parent,action or path)

                cache_val = frontierQueue.contains2(neighbor)
                if (neighbor not in explored_list) and (not cache_val):  # and (child_node not in frontierQueue):
                    # path_cost = cost_cum + child_node[0]
                    # path_cost |= graph[node_state][neighbor]['weight']
                    # temp_path = child_node[2][:]  # list(ucs_path)
                    # temp_path.append(neighbor)
                    frontierQueue.append(child_node)#[0],child_node[1],child_node[2],child_node[3]))
                elif(cache_val and frontierQueue.getCost(neighbor) > cost):
                    frontierQueue.remove2(neighbor)
                    frontierQueue.append(child_node)


            # graph.explored_nodes()
        return []

    # TODO: finish this function!
    #raise NotImplementedError


def null_heuristic(graph, v, goal):
    """
    Null heuristic used as a base line.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        0
    """

    return 0


def euclidean_dist_heuristic(graph, v, goal):
    """
    Warm-up exercise: Implement the euclidean distance heuristic.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Euclidean distance between `v` node and `goal` node
    """

    x1, y1 = graph.node[goal]['pos']
    x2, y2 = graph.node[v]['pos']

    x = (x2-x1)
    y = (y2-y1)

    dist = math.sqrt(x**2 + y**2)

    return dist

    # TODO: finish this function!
    #raise NotImplementedError


def a_star(graph, start, goal, heuristic=euclidean_dist_heuristic):
    """
    Warm-up exercise: Implement A* algorithm.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    if start == goal:
        return []
    else:
        path_cost = 0
        heuristic_cost = 0
        explored_list = set()
        frontierQueue = PriorityQueue()
        frontierQueue.append((heuristic_cost,start,path_cost,start,[start])) #node(h_cost,state,cost,parent,action or path)

        while frontierQueue.size() > 0:

            hcost,node_state,pcost,p_node,n_path = frontierQueue.pop()[1] #node(h_cost,cost,state,parent,action or path)
            # node_state = state

            if node_state == goal:
                return list(n_path) #list element of the dictionary

            explored_list.add(node_state)

            neighbor_nodes = graph[node_state]  # get the neighbors
            for neighbor in neighbor_nodes:
                h_cost = euclidean_dist_heuristic(graph, neighbor, goal)
                p_cost = graph[node_state][neighbor]['weight']+pcost
                f_h_cost = h_cost+p_cost
                node_path = n_path[:]
                node_path.append(neighbor)
                child_node = (f_h_cost,neighbor,p_cost,node_state,node_path) #node(cost,state,parent,action or path)

                cache_val = frontierQueue.contains2(neighbor)
                if (neighbor not in explored_list) and (not cache_val):  # and (child_node not in frontierQueue):
                    frontierQueue.append(child_node)#[0],child_node[1],child_node[2],child_node[3]))
                elif(cache_val and frontierQueue.getCost(neighbor) > f_h_cost):
                    frontierQueue.remove2(neighbor)
                    frontierQueue.append(child_node)

            # graph.explored_nodes()
        return []

    # TODO: finish this function!
    #raise NotImplementedError


def bidirectional_ucs(graph, start, goal):
    """
    Exercise 1: Bidirectional Search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    if start == goal:
        return []
    else: # global loop

        miu = float("inf")
        top_f = 0
        top_r = 0 #float("inf")

        path12 = PriorityQueue()
        pathAB = pathBA = False

        backwardPathQueue = PriorityQueue()
        forwardPathQueue = PriorityQueue()

        path_cost = 0
        explored_list = set()
        frontierQueue = PriorityQueue()
        n_path = start # unicode(start)
        frontierQueue.append((path_cost,n_path,n_path,[n_path])) #node(cost,state,parent,action or path)

        path_cost2 = 0
        explored_list2 = set()
        frontierQueue2 = PriorityQueue()
        n_path2 = goal # unicode(goal)
        frontierQueue2.append((path_cost2,n_path2,n_path2,[n_path2]))  # node(cost,state,parent,action or path)

        while (frontierQueue.size() > 0) or (frontierQueue2.size() > 0):
            # check if two paths have been found and return the cheapest
            if (top_f + top_r >= miu) and (pathBA or pathAB):
                cost_1, sub_path1, t_path_1 = path12.pop()[1]
                return list(t_path_1)

            node_checked1 = node_checked2 = False

            if frontierQueue.size() > 0:
                p_cost,node_state,p_node,n_path = frontierQueue.pop()[1] #node(cost,state,parent,action or path)
                forwardPathQueue.append((p_cost,node_state,n_path))
                explored_list.add(node_state)

                f1 = True
            else:
                f1 = False

            if frontierQueue2.size() > 0:
                p_cost2, node_state2, p_node2, n_path2 = frontierQueue2.pop()[1]  # node(cost,state,parent,action or path)
                backwardPathQueue.append((p_cost2, node_state2, n_path2))
                explored_list2.add(node_state2)

                f2 = True
            else:
                f2 = False

            # check if fwd path intercepts backward path
            if f1 and f2:
                neighbor_nodes = graph[node_state]  # get the neighbors
                for neighbor in neighbor_nodes:
                    cost = neighbor_nodes[neighbor]['weight'] + p_cost
                    node_path = n_path[:]
                    node_path.append(neighbor)
                    #check before exploration
                    if neighbor == goal:
                        node_checked1 = True
                        top_f = cost
                        if pathAB:
                            if path12.getCost("AB") > cost:
                                path12.remove2("AB")  # item[1][1]
                                path12.append((cost, "AB", node_path))
                                miu = cost
                        else:
                            pathAB = True
                            path12.append((cost, "AB", node_path))
                            miu = cost
                    elif neighbor in explored_list2:
                        node_checked1 = True
                        node_intersect = backwardPathQueue.getNode(neighbor)
                        path_intersect = node_intersect[1][-1]
                        n_path = n_path[:]
                        top_f = cost
                        scost = cost + node_intersect[1][0]  # - s_cost
                        if pathAB:
                            if path12.getCost("AB") > scost:
                                path12.remove2("AB")  # item[1][1]
                                path12.append((scost, "AB", list(n_path) + list(reversed(list(path_intersect)))))
                                miu = scost
                        else:
                            pathAB = True
                            path12.append((scost, "AB", list(n_path) + list(reversed(list(path_intersect)))))
                            miu = scost
                    elif neighbor == node_state2:
                        node_checked1 = True
                        n_path = n_path[:]
                        top_f = cost
                        scost = cost + p_cost2  # - s_cost
                        if pathAB:
                            if path12.getCost("AB") > scost:
                                path12.remove2("AB")  # item[1][1]
                                path12.append((scost, "AB", list(n_path) + list(reversed(list(n_path2)))))
                                miu = scost
                        else:
                            pathAB = True
                            path12.append((scost, "AB", list(n_path) + list(reversed(list(n_path2)))))
                            miu = scost
                    elif node_state == node_state2:
                        node_checked1 = True
                        n_path = n_path[:-1]
                        if p_cost > p_cost2:
                            top_f = p_cost
                        else:
                            top_f = p_cost2
                        scost = p_cost + p_cost2  # - s_cost
                        if pathAB:
                            if path12.getCost("AB") > scost:
                                path12.remove2("AB")  # item[1][1]
                                path12.append((scost, "AB", list(n_path) + list(reversed(list(n_path2)))))
                                miu = scost
                        else:
                            pathAB = True
                            path12.append((scost, "AB", list(n_path) + list(reversed(list(n_path2)))))
                            miu = scost

                    if (not node_checked1):
                        child_node = (cost, neighbor, node_state, node_path)  # node(cost,state,parent,action or path)
                        cache_val = frontierQueue.contains2(neighbor)

                        if (neighbor not in explored_list) and (not cache_val):
                            frontierQueue.append(child_node)
                        elif (cache_val and frontierQueue.getCost(neighbor) > cost):
                            frontierQueue.remove2(neighbor)
                            frontierQueue.append(child_node)
                    else:
                        forwardPathQueue.append((cost, neighbor, node_path))

                    # check if two paths have been found and return the cheapest
                    if (top_f + top_r >= miu) and (pathBA or pathAB):
                        cost_1, sub_path1, t_path_1 = path12.pop()[1]
                        return list(t_path_1)


                # backward search starts here===================================================
                neighbor_nodes2 = graph[node_state2]  # get the neighbors
                for neighbor2 in neighbor_nodes2:
                    cost2 = neighbor_nodes2[neighbor2]['weight'] + p_cost2
                    n_path2 = n_path2[:]
                    node_path2 = n_path2[:]
                    node_path2.append(neighbor2)

                    # check if backwrd path intercepts fwd path
                    if neighbor2 == start:
                        node_checked2 = True
                        top_r = cost2
                        if pathBA:
                            if path12.getCost("BA") > cost2:
                                path12.remove2("BA")  # item[1][1]
                                path12.append((cost2, "BA", list(reversed(list(node_path2)))))
                                miu = cost2
                        else:
                            pathBA = True
                            path12.append((p_cost2, "BA", list(reversed(list(node_path2)))))
                            miu = cost2
                    elif neighbor2 in explored_list:
                        node_checked2 = True
                        node_intersect2 = forwardPathQueue.getNode(neighbor2)
                        path_intersect2 = node_intersect2[1][-1]
                        n_path2 = n_path2[:]
                        top_r = cost2
                        scost = cost2 + node_intersect2[1][0]  # - s_cost
                        if pathBA:
                            if path12.getCost("BA") > scost:
                                path12.remove2("BA")  # item[1][1]
                                path12.append((scost, "BA", list(path_intersect2) + list(reversed(list(n_path2)))))
                                miu = scost
                        else:
                            pathBA = True
                            path12.append((scost, "BA", list(path_intersect2) + list(reversed(list(n_path2)))))
                            miu = scost
                    elif neighbor2 == node_state:
                        node_checked2 = True
                        n_path2 = n_path2[:]
                        top_r = cost2
                        scost = p_cost + cost2  # - s_cost
                        if pathBA:
                            if path12.getCost("BA") > scost:
                                path12.remove2("BA")  # item[1][1]
                                path12.append((scost, "BA", list(n_path) + list(reversed(list(n_path2)))))
                                miu = scost
                        else:
                            pathBA = True
                            path12.append((scost, "BA", list(n_path) + list(reversed(list(n_path2)))))
                            miu = scost

                    if (not node_checked2):
                        child_node2 = (cost2, neighbor2, node_state2, node_path2)  # node(cost,state,parent,action or path)
                        cache_val2 = frontierQueue2.contains2(neighbor2)

                        if (neighbor2 not in explored_list2) and (not cache_val2):  # and (child_node not in frontierQueue):
                            frontierQueue2.append(child_node2)  # [0],child_node[1],child_node[2],child_node[3]))
                        elif (cache_val2 and frontierQueue2.getCost(neighbor2) > cost2):
                            frontierQueue2.remove2(neighbor2)
                            frontierQueue2.append(child_node2)
                    else:
                        backwardPathQueue.append((cost2, neighbor2, node_path2))


            elif f1 and not f2:
                neighbor_nodes = graph[node_state]  # get the neighbors
                for neighbor in neighbor_nodes:
                    cost = neighbor_nodes[neighbor]['weight'] + p_cost
                    node_path = n_path[:]
                    node_path.append(neighbor)
                    # check before exploration
                    if neighbor == goal:
                        node_checked1 = True
                        top_f = cost
                        if pathAB:
                            if path12.getCost("AB") > cost:
                                path12.remove2("AB")  # item[1][1]
                                path12.append((cost, "AB", node_path))
                                miu = cost
                        else:
                            pathAB = True
                            path12.append((cost, "AB", node_path))
                            miu = cost
                    elif neighbor in explored_list2:
                        node_checked1 = True
                        node_intersect = backwardPathQueue.getNode(neighbor)
                        path_intersect = node_intersect[1][-1]
                        n_path = n_path[:]
                        top_f = cost
                        scost = cost + node_intersect[1][0]  # - s_cost
                        if pathAB:
                            if path12.getCost("AB") > scost:
                                path12.remove2("AB")  # item[1][1]
                                path12.append((scost, "AB", list(n_path) + list(reversed(list(path_intersect)))))
                                miu = scost
                        else:
                            pathAB = True
                            path12.append((scost, "AB", list(n_path) + list(reversed(list(path_intersect)))))
                            miu = scost

                    # elif neighbor == node_state2:
                    #     node_checked1 = True
                    #     n_path = n_path[:]
                    #     top_f = cost
                    #     scost = cost + p_cost2  # - s_cost
                    #     if pathAB:
                    #         if path12.getCost("AB") > scost:
                    #             path12.remove2("AB")  # item[1][1]
                    #             path12.append((scost, "AB", list(n_path) + list(reversed(list(n_path2)))))
                    #             miu = scost
                    #     else:
                    #         pathAB = True
                    #         path12.append((scost, "AB", list(n_path) + list(reversed(list(n_path2)))))
                    #         miu = scost
                    # elif node_state == node_state2:
                    #     node_checked1 = True
                    #     n_path = n_path[:-1]
                    #     if p_cost > p_cost2:
                    #         top_f = p_cost
                    #     else:
                    #         top_f = p_cost2
                    #     scost = p_cost + p_cost2  # - s_cost
                    #     if pathAB:
                    #         if path12.getCost("AB") > scost:
                    #             path12.remove2("AB")  # item[1][1]
                    #             path12.append((scost, "AB", list(n_path) + list(reversed(list(n_path2)))))
                    #             miu = scost
                    #     else:
                    #         pathAB = True
                    #         path12.append((scost, "AB", list(n_path) + list(reversed(list(n_path2)))))
                    #         miu = scost

                    if (not node_checked1):
                        child_node = (cost, neighbor, node_state, node_path)  # node(cost,state,parent,action or path)
                        cache_val = frontierQueue.contains2(neighbor)

                        if (neighbor not in explored_list) and (not cache_val):
                            frontierQueue.append(child_node)
                        elif (cache_val and frontierQueue.getCost(neighbor) > cost):
                            frontierQueue.remove2(neighbor)
                            frontierQueue.append(child_node)
                    else:
                        forwardPathQueue.append((cost, neighbor, node_path))


            elif f2 and not f1:
                # backward search starts here===================================================
                neighbor_nodes2 = graph[node_state2]  # get the neighbors
                for neighbor2 in neighbor_nodes2:
                    cost2 = neighbor_nodes2[neighbor2]['weight'] + p_cost2
                    n_path2 = n_path2[:]
                    node_path2 = n_path2[:]
                    node_path2.append(neighbor2)

                    # check if backwrd path intercepts fwd path
                    if neighbor2 == start:
                        node_checked2 = True
                        top_r = cost2
                        if pathBA:
                            if path12.getCost("BA") > cost2:
                                path12.remove2("BA")  # item[1][1]
                                path12.append((cost2, "BA", list(reversed(list(node_path2)))))
                                miu = cost2
                        else:
                            pathBA = True
                            path12.append((p_cost2, "BA", list(reversed(list(node_path2)))))
                            miu = cost2
                    elif neighbor2 in explored_list:
                        node_checked2 = True
                        node_intersect2 = forwardPathQueue.getNode(neighbor2)
                        path_intersect2 = node_intersect2[1][-1]
                        n_path2 = n_path2[:]
                        top_r = cost2
                        scost = cost2 + node_intersect2[1][0]  # - s_cost
                        if pathBA:
                            if path12.getCost("BA") > scost:
                                path12.remove2("BA")  # item[1][1]
                                path12.append((scost, "BA", list(path_intersect2) + list(reversed(list(n_path2)))))
                                miu = scost
                        else:
                            pathBA = True
                            path12.append((scost, "BA", list(path_intersect2) + list(reversed(list(n_path2)))))
                            miu = scost
                    # elif neighbor2 == node_state:
                    #     node_checked2 = True
                    #     n_path2 = n_path2[:]
                    #     top_r = cost2
                    #     scost = p_cost + cost2  # - s_cost
                    #     if pathBA:
                    #         if path12.getCost("BA") > scost:
                    #             path12.remove2("BA")  # item[1][1]
                    #             path12.append((scost, "BA", list(n_path) + list(reversed(list(n_path2)))))
                    #             miu = scost
                    #     else:
                    #         pathBA = True
                    #         path12.append((scost, "BA", list(n_path) + list(reversed(list(n_path2)))))
                    #         miu = scost

                    if (not node_checked2):
                        child_node2 = (
                            cost2, neighbor2, node_state2, node_path2)  # node(cost,state,parent,action or path)
                        cache_val2 = frontierQueue2.contains2(neighbor2)

                        if (neighbor2 not in explored_list2) and (
                                not cache_val2):  # and (child_node not in frontierQueue):
                            frontierQueue2.append(child_node2)  # [0],child_node[1],child_node[2],child_node[3]))
                        elif (cache_val2 and frontierQueue2.getCost(neighbor2) > cost2):
                            frontierQueue2.remove2(neighbor2)
                            frontierQueue2.append(child_node2)
                    else:
                        backwardPathQueue.append((cost2, neighbor2, node_path2))

        if path12.size() > 0:
            cost_1, sub_path1, t_path_1 = path12.pop()[1]
            return list(t_path_1)
        else:
            return []

    # TODO: finish this function!
    # raise NotImplementedError


def bidirectional_a_star(graph, start, goal, heuristic=euclidean_dist_heuristic):
    """
    Exercise 2: Bidirectional A*.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    if start == goal:
        return []
    else: # global loop
        backwardPathQueue = PriorityQueue()
        forwardPathQueue = PriorityQueue()

        path_cost = 0
        explored_list = set()
        heuristic_cost = 0
        frontierQueue = PriorityQueue()
        n_path = start # unicode(start)
        frontierQueue.append((heuristic_cost,n_path,path_cost,n_path,[n_path])) #node(h_cost,state,cost,parent,action or path)

        path_cost2 = 0
        explored_list2 = set()
        heuristic_cost2 = 0
        frontierQueue2 = PriorityQueue()
        n_path2 = goal # unicode(goal)
        frontierQueue2.append((heuristic_cost2,n_path2,path_cost2,n_path2,[n_path2]))  # node(cost,state,parent,action or path)

        while (frontierQueue.size() > 0) and (frontierQueue2.size() > 0):
            # if frontierQueue.size() > 0:
            hcost, node_state, pcost, p_node, n_path = frontierQueue.pop()[1] #node(cost,state,parent,action or path)
            forwardPathQueue.append((hcost,node_state,n_path))

            hcost2, node_state2, pcost2, p_node2, n_path2 = frontierQueue2.pop()[1]  # node(cost,state,parent,action or path)
            backwardPathQueue.append((hcost2,node_state2,n_path2))

            if node_state == goal:
                return list(n_path)
            elif node_state in explored_list2:
                node_intersect = backwardPathQueue.getNode(node_state)
                path_intersect = node_intersect[1][-1]
                n_path = n_path[:-1]
                return list(n_path) + list(reversed(path_intersect)) #list element of the dictionary
            elif node_state == node_state2:
                n_path2 = n_path2[:-1]
                return list(n_path) + list(reversed(n_path2))#list element of the dictionary

            #forward search starts here===========================================
            explored_list.add(node_state)

            neighbor_nodes = graph[node_state]  # get the neighbors
            for neighbor in neighbor_nodes:
                h_cost = euclidean_dist_heuristic(graph, neighbor, goal)
                p_cost = neighbor_nodes[neighbor]['weight']+pcost
                f_h_cost = h_cost + p_cost
                node_path = n_path[:]
                node_path.append(neighbor)
                child_node = (f_h_cost,neighbor,p_cost,node_state,node_path)

                cache_val = frontierQueue.contains2(neighbor)
                if (neighbor not in explored_list) and (not cache_val):
                    frontierQueue.append(child_node)
                elif(cache_val and frontierQueue.getCost(neighbor) > f_h_cost):
                    frontierQueue.remove2(neighbor)
                    frontierQueue.append(child_node)
            #forward search ends here===========================================

            if node_state2 == start:
                return list(reversed(n_path2))
            elif node_state2 in explored_list:
                node_intersect2 = forwardPathQueue.getNode(node_state2)
                path_intersect2 = node_intersect2[1][-1]
                n_path2 = n_path2[:-1]
                return list(path_intersect2) + list(reversed(n_path2))  # list element of the dictionary
            elif node_state2 == node_state:
                n_path2 = n_path2[:-1]
                return list(n_path) + list(reversed(n_path2))#list element of the dictionary

            # backward search starts here===================================================
            explored_list2.add(node_state2)

            neighbor_nodes2 = graph[node_state2]  # get the neighbors
            for neighbor2 in neighbor_nodes2:
                h_cost2 = euclidean_dist_heuristic(graph, neighbor2, start)
                p_cost2 = neighbor_nodes2[neighbor2]['weight'] + pcost2
                f_h_cost2 = h_cost2 + p_cost2
                node_path2 = n_path2[:]
                node_path2.append(neighbor2)
                child_node2 = (f_h_cost2, neighbor2, p_cost2, node_state2, node_path2)

                cache_val2 = frontierQueue2.contains2(neighbor2)
                if (neighbor2 not in explored_list2) and (not cache_val2):
                    frontierQueue2.append(child_node2)
                elif (cache_val2 and frontierQueue2.getCost(neighbor2) > f_h_cost2):
                    frontierQueue2.remove2(neighbor2)
                    frontierQueue2.append(child_node2)
            # backward search ends here===================================================

            if node_state2 == start:
                return list(reversed(n_path2))
            elif node_state2 in explored_list:
                node_intersect2 = forwardPathQueue.getNode(node_state2)
                path_intersect2 = node_intersect2[1][-1]
                n_path2 = n_path2[:-1]
                return list(path_intersect2) + list(reversed(n_path2))  # list element of the dictionary
            elif node_state2 == node_state:
                n_path2 = n_path2[:-1]
                return list(n_path) + list(reversed(n_path2))  # list element of the dictionary

        return []

    # TODO: finish this function!
    # raise NotImplementedError


def sum_weight(graph, path):
    """
    Calculate the total cost of a path by summing edge weights.

    Args:
        graph (ExplorableGraph): Graph that contains path.
        path (list(nodes)): List of nodes from src to dst.

    Returns:
        Sum of edge weights in path.
    """
    pairs = zip(path, path[1:])

    return sum([graph.get_edge_data(a, b)['weight'] for a, b in pairs])


def getPath_ucs(graph,goal_a,goal_b):

    return bidirectional_ucs(graph, goal_a, goal_b)


def getPath_astar(graph,goal_a,goal_b):

    return bidirectional_a_star(graph, goal_a, goal_b)


def tridirectional_search(graph, goals):
    """
    Exercise 3: Tridirectional UCS Search

    See README.MD for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """

    goal1 = goals[0]
    goal2 = goals[1]
    goal3 = goals[2]

    if goal1 == goal2 == goal3:
        return []
    else: # global loop
        path123 = PriorityQueue()

        path_AB = getPath_ucs(graph,goal1,goal2)
        path_BC = getPath_ucs(graph,goal2,goal3)
        path_AC = getPath_ucs(graph,goal1,goal3)

        path123.append((sum_weight(graph,path_AB), "AB", path_AB))
        path123.append((sum_weight(graph, path_BC), "BC", path_BC))
        path123.append((sum_weight(graph, path_AC), "AC", path_AC))

        cost_1, sub_path1, t_path_1 = path123.pop()[1]
        cost_2, sub_path2, t_path_2 = path123.pop()[1]

        if sub_path1 == "AB" and sub_path2 == "BC": #ABC
            if goal3 in t_path_1:
                return t_path_1
            elif goal1 in t_path_2:
                return t_path_2
            else:
                return list(t_path_1[:-1]) + list(t_path_2)

        elif sub_path1 == "BC" and sub_path2 == "AB": #ABC
            if goal1 in t_path_1:
                return t_path_1
            elif goal3 in t_path_2:
                return t_path_2
            else:
                return list(t_path_2[:-1]) + list(t_path_1)


        elif sub_path1 == "AC" and sub_path2 == "BC": #ACB
            if goal2 in t_path_1:
                return t_path_1
            elif goal1 in t_path_2:
                return t_path_2
            else:
                return list(t_path_1[:-1]) + list(reversed(list(t_path_2)))

        elif sub_path1 == "BC" and sub_path2 == "AC": #ACB
            if goal1 in t_path_1:
                return t_path_1
            elif goal2 in t_path_2:
                return t_path_2
            else:
                return list(t_path_2[:-1]) + list(reversed(list(t_path_1)))


        elif sub_path1 == "AC" and sub_path2 == "AB": #CAB
            if goal2 in t_path_1:
                return t_path_1
            elif goal3 in t_path_2:
                return t_path_2
            else:
                t_path0 = list(reversed(list(t_path_1)))
                t_path0 = t_path0[:-1]
                return list(t_path0) + list(t_path_2)

        elif sub_path1 == "AB" and sub_path2 == "AC":
            if goal3 in t_path_1:
                return t_path_1
            elif goal2 in t_path_2:
                return t_path_2
            else:
                t_path0 = list(reversed(list(t_path_2)))
                t_path0 = t_path0[:-1]
                return list(t_path0) + list(t_path_1)

    return []

    # TODO: finish this function
    # raise NotImplementedError


def tridirectional_upgraded(graph, goals, heuristic=euclidean_dist_heuristic):
    """
    Exercise 3: Upgraded Tridirectional Search

    See README.MD for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """

    goal1 = goals[0]
    goal2 = goals[1]
    goal3 = goals[2]

    if goal1 == goal2 == goal3:
        return []
    else: # global loop
        path123 = PriorityQueue()

        path_AB = getPath_astar(graph,goal1,goal2)
        path_BC = getPath_astar(graph,goal2,goal3)
        path_AC = getPath_astar(graph,goal1,goal3)

        path123.append((sum_weight(graph,path_AB), "AB", path_AB))
        path123.append((sum_weight(graph, path_BC), "BC", path_BC))
        path123.append((sum_weight(graph, path_AC), "AC", path_AC))

        cost_1, sub_path1, t_path_1 = path123.pop()[1]
        cost_2, sub_path2, t_path_2 = path123.pop()[1]

        if sub_path1 == "AB" and sub_path2 == "BC": #ABC
            if goal3 in t_path_1:
                return t_path_1
            elif goal1 in t_path_2:
                return t_path_2
            else:
                return list(t_path_1[:-1]) + list(t_path_2)

        elif sub_path1 == "BC" and sub_path2 == "AB": #ABC
            if goal1 in t_path_1:
                return t_path_1
            elif goal3 in t_path_2:
                return t_path_2
            else:
                return list(t_path_2[:-1]) + list(t_path_1)


        elif sub_path1 == "AC" and sub_path2 == "BC": #ACB
            if goal2 in t_path_1:
                return t_path_1
            elif goal1 in t_path_2:
                return t_path_2
            else:
                return list(t_path_1[:-1]) + list(reversed(list(t_path_2)))

        elif sub_path1 == "BC" and sub_path2 == "AC": #ACB
            if goal1 in t_path_1:
                return t_path_1
            elif goal2 in t_path_2:
                return t_path_2
            else:
                return list(t_path_2[:-1]) + list(reversed(list(t_path_1)))


        elif sub_path1 == "AC" and sub_path2 == "AB": #CAB
            if goal2 in t_path_1:
                return t_path_1
            elif goal3 in t_path_2:
                return t_path_2
            else:
                t_path0 = list(reversed(list(t_path_1)))
                t_path0 = t_path0[:-1]
                return list(t_path0) + list(t_path_2)

        elif sub_path1 == "AB" and sub_path2 == "AC":
            if goal3 in t_path_1:
                return t_path_1
            elif goal2 in t_path_2:
                return t_path_2
            else:
                t_path0 = list(reversed(list(t_path_2)))
                t_path0 = t_path0[:-1]
                return list(t_path0) + list(t_path_1)

    return []

    # TODO: finish this function
    #raise NotImplementedError


def return_your_name():
    """Return your name from this function"""

    return "Kehinde Aina"

    # TODO: finish this function
    # raise NotImplementedError


# Extra Credit: Your best search method for the race
def custom_search(graph, start, goal, data=None):
    """
    Race!: Implement your best search algorithm here to compete against the
    other student agents.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        data :  Data used in the custom search.
            Default: None.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    raise NotImplementedError


def load_data():
    """
    Loads data from data.pickle and return the data object that is passed to
    the custom_search method.

    Will be called only once. Feel free to modify.

    Returns:
         The data loaded from the pickle file.
    """

    dir_name = os.path.dirname(os.path.realpath(__file__))
    pickle_file_path = os.path.join(dir_name, "data.pickle")
    data = pickle.load(open(pickle_file_path, 'rb'))
    return data
