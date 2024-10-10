
# Q1
def function1 (nestedlist,fitmons):
    """
        This function used to fuse fitmon
        Written by Chong Yi Ming
        Precondition: cuteness_score is the fusion of left and right parent fitmons from fitmons[1...fused_level-1]
        Postcondition: cuteness_score is the fusion of left and right parent fitmons from fitmons[1...fusedlevel]
        Input: nestedlist(matrix),fitmons (list of fitmons)
        Return: nestedlist[0][len(nestedlist) - 1] (final cuteness_score)
        Time complexity:O(n^2)+O(n^3)
        Best case analysis:O(n^3)
        Worst case analysis:O(n^3)
        Space complexity:O(n^2)
        Input space analysis: nestedlist=O(N^2),fitmons=O(N)
        Aux space analysis:O(1)
        """
    fusedlevel=2
    while fusedlevel<len(nestedlist)+1:#indicates the level of fusion,start from 2 to len(fitmons)/len(nestedlist)
        for start in range(len(nestedlist) - fusedlevel + 1): # loop from range from 0 to len(nestedlist) - fusedlevel + 1,this is to find the left lower bound of the left group of fitmon in the current level of fusion
            subprob_highest_score = 0 #store the highest subprob combination cuteness score
            for i in range(start, start + fusedlevel - 1): #i is  to control the right upper bound of the left group of fitmons/fitmon while start+fusedlevel-1 is to control the right upper bound of right group of fitmons/fitmon on the current level of fusion
                fusion_cuteness = nestedlist[start][i] * fitmons[i][2] + nestedlist[i + 1][start + fusedlevel - 1] * fitmons[i + 1][0] #count the cuteness of the fused of left and right group of fitmons/fitmon and store the score
                if fusion_cuteness>subprob_highest_score:#it will store the highest cuteness score of fitmons from the range that specify, for example if we want to fuse from 0 to 3, the left group of fitmons might be index 0 to 1 of fitmon and right group might be index 2 to 3 of fitmon or left group is index 0 to 2 and right group is only index 3 and more, but it will only store thr highest score from index 0 to 3
                    subprob_highest_score=fusion_cuteness
            nestedlist[start][start + fusedlevel - 1] = int(subprob_highest_score) #assign the highest score into the matrix so this value wil being used if the range is being used to fuse with other fitmons
        fusedlevel+=1 #go to next level of fusion
    return nestedlist[0][len(nestedlist) - 1]#return the final cuteness score
def fuse(fitmons):
    """
    This function used to create matrix and call fuse function
    Written by Chong Yi Ming
    Precondition:-
    Postcondition:-
    Input: fitmons (list of fitmons)
    Return:cuteness_score is the answer of final fusion fitmon
    Time complexity:O(N^2)+
    Best case analysis:O(N)
    Worst case analysis:O(N)
    Space complexity:O(N^2)
    Input space analysis: O(N)
    Aux space analysis:O(N^2)
    """
    answer = []
    fitmonslist = len(fitmons)
    """
    build matrices and insert fitmon into the matrices 
    """
    for index in range(fitmonslist):
        answer.append([0] * fitmonslist)
        answer[index][index] = fitmons[index][1]
        #store the cuteness score for each fitmons into the matrix
    cuteness_score = function1(answer,fitmons)
    return cuteness_score


# ==========
# Q2



class PriorityQueue(object):
    """
    Reference: https://www.geeksforgeeks.org/priority-queue-in-python/
    """
    def __init__(self):
        """
        This init function used to initialise a list that will be used as priority queue since it has priority queue function
        Written by Chong Yi Ming
        Precondition:
        Postcondition:
        Input: -
        Return:None
        Time complexity:O(1)
        Best case analysis:O(1)
        Worst case analysis:O(1)
        Space complexity:O(1)
        Input space analysis: O(1)
        Aux space analysis:O(1)
        """
        self.list = []

    # def __str__(self):
    #     return ' '.join([str(i) for i in self.list])

    # for checking if the list is empty
    def isEmpty(self):
        """
            This isEmpty function used to check if self.list contain any element
            Written by Chong Yi Ming
            Precondition:
            Postcondition:
            Input: -
            Return: Return yes if there is element in self.list else return no
            Time complexity:O(1)
            Best case analysis:O(1)
            Worst case analysis:O(1)
            Space complexity:O(1)
            Input space analysis: O(1)
            Aux space analysis:O(1)
            """
        return len(self.list) == 0

    def __len__(self):
        """
        This __len__ function used to return the number of element in self.list
        Written by Chong Yi Ming
        Precondition:
        Postcondition:
        Input: -
        Return: Return number of element in the self.list
        Time complexity:O(1)
        Best case analysis:O(1)
        Worst case analysis:O(1)
        Space complexity:O(1)
        Input space analysis: O(1)
        Aux space analysis:O(1)
        """
        return len(self.list)

    # for inserting an element in the list
    def insert(self, data, priority):
        """
        This insert function used to append the tree with the amount of time as tuple into into self.list
        Written by Chong Yi Ming
        Precondition:
        Postcondition:
        Input: data(Tree object),priority(amount of time from u to v)
        Return: -
        Time complexity:O(1)
        Best case analysis:O(1)
        Worst case analysis:O(1)
        Space complexity:O(1)
        Input space analysis: O(1)
        Aux space analysis:O(1)
        """
        self.list.append((data, priority))


    def delete(self):
        """
        This delete function used to delete the high priority value item
        Written by Chong Yi Ming
        Precondition:
        Postcondition:
        Input: -
        Return: item(highest priority item)
        Time complexity:O(N)
        Best case analysis:O(N), N is length of self.list
        Worst case analysis:O(N)
        Space complexity:O(1)
        Input space analysis: O(1)
        Aux space analysis:O(1)
        """
        highest_val = 0
        for i in range(len(self.list)):
            if self.list[i][1] < self.list[highest_val][1]: #find if there is higher priority value
                highest_val = i
        maxitem = self.list[highest_val]
        del self.list[highest_val] #delete the item once we found the highest priority
        return maxitem

    def update(self, item, distance):
        """
        This update function used to remove and insert the modified amount of time (priority)back to the list
        Written by Chong Yi Ming
        Precondition:
        Postcondition:
        Input: item(Tree object),distance(amount of time from u to v)
        Return: -
        Time complexity:O(N)
        Best case analysis:O(N), N is length of self.list
        Worst case analysis:O(N)
        Space complexity:O(1)
        Input space analysis: O(1)
        Aux space analysis:O(1)
        """
        while not self.isEmpty():
            current_item, priority = self.delete()
            if current_item == item:
                current_item.distance = distance
                self.insert(item, distance)  # Update the distance of the target item
                break
            else:
                self.insert(item, priority)


class TreeMap:

    def __init__(self, roads, solulus):
        """
        This init function used to initialise array and and reversed array that store treeobject
        Written by Chong Yi Ming
        Precondition:
        Postcondition:
        Input: roads (list of roads), solulus(list of solulus tree)
        Return:None
        Time complexity:O(N^2)+
        Best case analysis:O(N)
        Worst case analysis:O(N)
        Space complexity:O(N^2)
        Input space analysis: O(R),O(S);R=length of roads,S=length of solulus
        Aux space analysis:O(N^2)+O(N)=O(N^2)
        """

        max_roads_0 = max(road[0] for road in roads)
        max_roads_1 = max(road[1] for road in roads)
        max_solulus_0 = max(solulu[0] for solulu in solulus)
        max_solulus_2 = max(solulu[2] for solulu in solulus)
        #get higest value from vertex and also length from roads and solulus list so that it wont output index out of bound
        self.highest_value = max(max_roads_0, max_roads_1, max_solulus_0, max_solulus_2,len(roads),len(solulus))
        self.vertices = [None] * ((self.highest_value)) #Normal Treemap
        self.secondvertices =[None] * ((self.highest_value)) #Reversed treemap
        self.solulustree=solulus
        self.solulusnode = [] #store all solulu trees' vertex
        self.solulusteleportnode=[] #store the teleported solulu trees' vertex
        for i in range(len(roads)): #construct the Treemap object
            u, v, w = roads[i]
            if self.vertices[u] is None:
                self.vertices[u] = Tree(u)
            if self.vertices[v] is None:
                self.vertices[v] = Tree(v)
            current_edge = Edge(u, v, w)
            self.vertices[u].add_edge(current_edge)

        for j in range(len(solulus)): #use to store value into solulusnode and solulusteleportnode
            u, w, v = solulus[j]
            self.solulusnode.append(u)
            self.solulusteleportnode.append(v)

        for i in range(len(roads)): #construct the reversed Treemap object
            v,u,w = roads[i]
            if self.secondvertices[v] is None:
                self.secondvertices[v] = Tree(v)
            if self.secondvertices[u] is None:
                self.secondvertices[u] = Tree(u)
            current_edge = Edge(u, v, w)
            self.secondvertices[u].add_edge(current_edge)



    def escape(self, start, exits):
        """
        This escape function used to call dijkstra function and merge the possible path between first treemap and second reversed treemap together and return distance and the path
        Written by Chong Yi Ming
        Precondition:
        Postcondition:
        Input: start (a start node), exits (list of exit nodes)
        Return: ((minvalue, minpath_int)); minvalue=the amount of time from start to any exits, minpath_int= the shortest path from start to any exits
        Time complexity:O(N^2)+
        Best case analysis:O(N)
        Worst case analysis:O(N)
        Space complexity:O(N^2)
        Input space analysis: start=O(1), exits=O(E);E= length of list of exit nodes
        Aux space analysis:O(N^2)+O(N)=O(N^2)
        """
        combinationshortestpath= self.dijkstra(start, self.solulusnode,0) #run dijkstra and get all tree objects that are solulus tree and the amount of time in a tuple
        firstgraphpath=[]

        #if we found the tree.id is equal to one of the solulus trees, we will backtrack its previous tree and get the shortest path from start to that specific solulus and append it to firstgraphpath
        for node in combinationshortestpath:
            for i in self.solulustree:
                current = node[0]
                if i[0]==current.id:
                    path=str(i[2])
                    path+=str(node[0].id)
                    while current.previous.id!=1:
                        path+=str(current.previous.id)
                        current = current.previous
                    path += str(start)
                    firstgraphpath.append((path,node[1]))

        #create a additional vertex xthat linked from the additional node to every exits in reveresed treemap
        for j in exits:
            if self.secondvertices[self.highest_value-1] is None:
                self.secondvertices[self.highest_value-1] = Tree(self.highest_value-1)
            current_edge=Edge(self.highest_value,j,0)
            self.secondvertices[self.highest_value-1].add_edge(current_edge)


        secondgraphpath=[]
        #assign the reverse treemap into self.vertices and run dijkstra to find every nodes that are teleported from solulus from the additional node that created
        self.vertices = self.secondvertices
        combinationshortestpath2 = self.dijkstratest(self.highest_value-1, self.solulusteleportnode,2)
        #backtrack and find the shortest path for each nodes that we found earlier and their amount of time travel into a list,secondgraphpath
        for node in combinationshortestpath2:
            for i in self.solulustree:
                current = node[0]
                if i[2] == current.id:
                    path=str(current.id)
                    while current.previous is not None and current.previous.id != self.highest_value-1:
                        path += str(current.previous.id)
                        current = current.previous
                    secondgraphpath.append((path, node[1]))

        #we need to flip the path that we get from first Treemap since it is actually reversed, and we have to make it start from start to solulus tree node
        for i in range(len(firstgraphpath)):

            path, distance = firstgraphpath[i]

            # Reverse the path from left to right
            reversed_path = path[::-1]

            # Remove consecutive same characters from the reversed path
            cleaned_path = ''
            prev_char = ''
            for char in reversed_path:
                if char != prev_char:
                    cleaned_path += char
                prev_char = char

            firstgraphpath[i] = (cleaned_path, distance)




        merged_list = []
        #we need to merge the first treemap and second reversed treemap so we can get shortest time from start to exits and plus the time at first treemap and second reversed treemap and append to merged_list
        #we need to check if the last node id part of exits nodes, never add to merged list if it's not
        for i in range(len(firstgraphpath)):
            for j in range(len(secondgraphpath)):
                # Check if the last character of the first element in firstlist
                # matches the first character of the first element in secondlist
                if firstgraphpath[i][0][-1] == secondgraphpath[j][0][0] and int(secondgraphpath[j][0][-1]) in exits:
                    # Append a tuple containing the combined path, combined distance, and the index of the pair
                    merged_list.append(
                        (firstgraphpath[i][0] + secondgraphpath[j][0][1:], firstgraphpath[i][1] + secondgraphpath[j][1]))

        #we need to get the minimum distance and convert the minpath from string to int and store in list minpath_int
        minvalue = merged_list[0][1]
        minpath = merged_list[0][0]
        for i in range(len(merged_list)):
            if int(merged_list[i][1]) < int(minvalue):
                minvalue = merged_list[i][1]
                minpath = merged_list[i][0]  # Update minpath here, not minvalue
        minpath_int = [int(char) for char in minpath]
        return ((minvalue, minpath_int))

    #used for start to solulus
    def dijkstra(self, source, destination,idx):

        #initialise the array for storing shortest path to destination
        source = self.vertices[source]
        combination_list = []
        distance = [float('inf')] * len(destination)  # Initialize distances with infinity

        visited_min_nodes = []  # Store visited minimum nodes and their distances
        #initialise priority queue
        self.discovered = PriorityQueue()
        self.discovered.insert(source, source.distance)
        # when there is still tuple in discovered that is (tree object,amount of time)
        while len(self.discovered) > 0:
            #get the high priority item
            u = self.discovered.delete()[0]
            visited_min_nodes.append((u, u.distance))  # Store visited minimum node into storing list (visitied min_nodes)
            while u.visited:  # Skip nodes that have already been visited else it will added the same distance another time, if node 2 to 5 is 2, if node 5 to 6 is 5, then when i want to reach from 2 to 6 will be 5+2 and it will add additional 2
                u = self.discovered.delete()[0]
                visited_min_nodes.append((u, u.distance))
            u.visited = True
            #check if the nodes reach the destination and we will store it in distance array
            if u.id in destination:
                # Store the shortest distance for this destination
                index = destination.index(u.id)
                if u.distance < distance[index]:
                    distance[index] = u.distance
                    for i in self.solulustree:
                        if i[idx]==u.id:
                            combination_list.append((u,distance[index]+i[1]))
                    # Reinsert visited minimum nodes into the priority list else we might not be  able to backtrack or get the previous shortest path
                    for i in visited_min_nodes:
                        self.discovered.insert(i[0], i[1])

            #when there is still outgoing edges in the node
            for edge in u.edges:
                v = edge.v
                v = self.vertices[v]
                #check if the node that linked to the outgoing edges is discovered, if no, then update the new distance
                if not v.discovered:
                    v.discovered = True
                    v.distance = u.distance + edge.w
                    self.discovered.insert(v, v.distance)
                    v.previous = u
                # check if the node that linked to the outgoing edges is visited, if no, and check if there is shortest path that go to this node and update the distance to shorter amount of time

                elif not v.visited:
                    if v.distance > u.distance + edge.w:
                        distance = u.distance + edge.w
                        self.discovered.update(v, distance)
                        v.previous = u
            #remove tuple that contain (tree object,amount of time) if every destination being found
            if len(combination_list)==len(destination):
                return combination_list
        return combination_list

    # used for exit to solulus teleported nodes
    def dijkstratest(self, source, destination, idx):
        # initialise the array for storing shortest path to destination
        source = self.vertices[source]
        combination_list = []
        distance = [float('inf')] * len(destination)


        visited_min_nodes = []  # Store visited minimum nodes and their distances
        self.discovered = PriorityQueue()
        self.discovered.insert(source, source.distance)

        while len(self.discovered) > 0:
            u = self.discovered.delete()[0]

            while u.visited:  # # Skip nodes that have already been visited else it will added the shortest previous distance if we traverse through the road distance another time again and again
                u = self.discovered.delete()[0]
            u.visited = True


            if u.id in destination:
                # Store the shortest distance for this destination
                index = destination.index(u.id)
                if u.distance < distance[index]:
                    distance[index] = u.distance
                    for i in self.solulustree:
                        if i[idx] == u.id:
                            combination_list.append((u, distance[index]))
                            visited_min_nodes.append(u.id)

            for edge in u.edges:
                v = edge.v
                v = self.vertices[v]
                if not v.discovered:
                    v.discovered = True
                    v.distance = u.distance + edge.w
                    self.discovered.insert(v, v.distance)
                    v.previous = u
                elif not v.visited:
                    if v.distance > u.distance + edge.w:
                        v.distance = u.distance + edge.w
                        self.discovered.update(v, v.distance)
                        v.previous = u

        # Add solulu teleported nodes connected to the additional with distance 0 if not visited before
        for soluluteleport in destination:
            if soluluteleport != self.highest_value and soluluteleport in self.solulusteleportnode:
                if soluluteleport not in visited_min_nodes:
                    combination_list.append((self.vertices[soluluteleport], 0))
                    visited_min_nodes.append(soluluteleport)
        return combination_list



class Tree:
    def __init__(self, id):
        """
        This __init__ function used to initialise the Tree object
        Written by Chong Yi Ming
        Precondition:
        Postcondition:
        Input: id(vertex number)
        Return: -
        Time complexity:O(1)
        Best case analysis:O(1)
        Worst case analysis:O(1)
        Space complexity:O(1)
        Input space analysis: O(1)
        Aux space analysis:O(1)
        """
        self.id = id
        self.edges = []
        self.discovered = False
        self.visited = False
        self.distance = 0
        self.previous = None



    def add_edge(self, edge):
        """
        This add_edge function used to add edges to the Tree object itself
        Written by Chong Yi Ming
        Precondition:
        Postcondition:
        Input: edge(Edge object)
        Return: -
        Time complexity:O(1)
        Best case analysis:O(1)
        Worst case analysis:O(1)
        Space complexity:O(1)
        Input space analysis: O(1)
        Aux space analysis:O(1)
        """
        self.edges.append(edge)


class Edge:

    def __init__(self, u, v, w):
        """
        This init function used to initialise Roads from starting Tree (u), ending tree(v) and the amount of time(w) from u to v
        Written by Chong Yi Ming
        Precondition:
        Postcondition:
        Input: u (Startig tree), v (ending tree), w (amount of time)
        Return:None
        Time complexity:O(1)
        Best case analysis:O(1)
        Worst case analysis:O(1)
        Space complexity:O(1)
        Input space analysis: O(1)
        Aux space analysis:O(1)
        """
        self.u = u
        self.v = v
        self.w = w
