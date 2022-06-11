from processImage import *
import cv2, numpy as np, time


class PriorityQueue:
    def __init__(self):
        self.list = []
        self.length = 0

    def compare(self, index1, index2):
        """
            Compare with respect to weight
            index1.weight < index1.weight
        """
        if index1 < 0 or index2 < 0:
            return False
        if index1 >= self.length or index2 >= self.length:
            return False
        return self.list[index1].weight < self.list[index2].weight

    def extract_min(self):
        assert self.length > 0, "EmptyQueue"
        minValue = self.list[0]
        self.length -= 1
        if self.length > 0:
            self.list[0] = self.list.pop(-1)
            self.heapify(0)
        else:
            minValue = self.list.pop(0)
        return minValue

    def insert(self, element):
        self.list.append(element)
        self.length += 1
        index = self.length - 1
        while index >= 0:
            parentIndex = (index - 1) // 2
            if self.compare(index, parentIndex):
                self.swap(index, parentIndex)
                index = parentIndex
            else:
                break

    def decreaseWeight(self, element):
        """
            TODO: Improve complexity: O(n)
            Find element in queue:
            If found:
                Set weight to minimum of two
                return True
            Else
                return False
        """
        for index, node in enumerate(self.list):
            if (node.x, node.y) == (element.x, element.y):
                if element.weight < node.weight:
                    self.list[index] = element
                    while index >= 0:
                        parentIndex = (index - 1) // 2
                        if self.compare(index, parentIndex):
                            self.swap(index, parentIndex)
                            index = parentIndex
                        else:
                            break
                return True
        return False

    def swap(self, index1, index2):
        self.list[index1], self.list[index2] = self.list[index2], self.list[index1]

    def heapify(self, index):
        while index < self.length:
            left = 2 * index + 1
            minIndex = left
            right = left + 1  # 2*index+2
            if right < self.length and self.compare(right, left):
                minIndex = right
            if self.compare(minIndex, index):
                # Swap
                self.swap(minIndex, index)
                index = minIndex
            else:
                break


class Node:
    def __init__(self, x, y, via, weight):
        self.x = x
        self.y = y
        self.via = via
        self.weight = weight

    def __eq__(self, other):
        return (self.x, self.y) == (other.x, other.y)

    def __hash__(self):
        return hash((self.x, self.y))

    def __repr__(self):
        return f"X = {self.x}, Y = {self.y}, via = ({self.via.x},{self.via.y}), weight = {self.weight}"


class MazeSolve:
    def __init__(
        self,
        fileName,
        visualization=False,
        visualization_speed=50,
        preProcessImage=True,
    ):
        """
            TODO: Generalize method for finding src and dst
            - Read image,
            - Store current size in self.size
            - Preprocess the image
            - Find src, dst (Only white in first and last row)
        """
        self.visualization = visualization
        self.visualization_speed = visualization_speed
        img = cv2.imread(fileName)
        self.size = img.shape[:2]
        if preProcessImage:
            img = preProcess(img)
        self.src = (np.where(img[0] == [255, 255, 255])[0][0], 0)
        self.dst = (
            np.where(img[img.shape[0] - 1] == [255, 255, 255])[0][0],
            img.shape[0] - 1,
        )
        self.img = img

    def getAdjacent(self, node):
        """
            Find adjacent nodes (4 connected)
            Filter adjacent based on image boundary
        """
        x_max, y_max = self.img.shape[:2]
        retList = []
        sign = 1
        for _ in range(2):
            points = [(node.x + 0, node.y + sign), (node.x + sign, node.y + 0)]
            for x, y in points:
                if (x >= 0 and x < x_max) and (y >= 0 and y < y_max):
                    retList.append((x, y))
            sign *= -1
        return retList

    def showFrame(self, node):
        if self.visualization:
            cv2.imshow("Frame", self.img)
            if cv2.waitKey(1) == 27:
                cv2.destroyAllWindows()
                exit(0)

    def heuristic(self, x, y, dst):
        """
            Find Euclidean distance between (x,y) and dst
        """
        return (dst[0] - x) ** 2 + (dst[1] - y) ** 2

    def Dijakstra(self):
        """
            Apply Dijakstra search algorithm
            Same as Astar, except h(x) is always 0
        """
        # ===== Init =====
        queue = PriorityQueue()
        node = Node(*self.src, Node(-1, -1, -1, -1), 0)
        visited = set()
        queue.insert(node)
        iteration = 0

        # ===== Algorithm =====
        while (node.x, node.y) != self.dst:
            node = queue.extract_min()
            if node in visited:
                continue
            if self.visualization:
                self.img[node.y, node.x] = [255, 0, 0]
            if iteration % self.visualization_speed == 0:
                self.showFrame(node)
            iteration += 1
            for x, y in self.getAdjacent(node):
                # Check if (x,y) is white node
                if np.array_equal(self.img[y, x], [255, 255, 255]):
                    tmp = Node(x, y, node, iteration)
                    # Check if node exists in queue, if it does set minimum possible weight
                    if not queue.decreaseWeight(tmp):
                        # Else add node to queue
                        queue.insert(tmp)
            visited.add(node)

        # ===== Retrace path =====
        while (node.x, node.y) != (-1, -1):
            self.img[node.y, node.x] = [0, 0, 255]
            node = node.via

        # ===== Write Image =====
        return self.write()

    def Astar(self):
        """
            Apply Astar search algorithm
            w(x) = g(x) + h(x)
            where
                - w(x) = Weight for node x
                - g(x) = Iteration
                - h(x) = Euclidian distance
        """
        # ===== Init =====
        queue = PriorityQueue()
        node = Node(*self.src, Node(-1, -1, -1, -1), 0)
        visited = set()
        queue.insert(node)
        iteration = 0

        # ===== Algorithm =====
        while (node.x, node.y) != self.dst:
            node = queue.extract_min()
            if node in visited:
                continue
            if self.visualization:
                self.img[node.y, node.x] = [255, 0, 0]
            if iteration % self.visualization_speed == 0:
                self.showFrame(node)
            iteration += 1
            for x, y in self.getAdjacent(node):
                # Check if (x,y) is white node
                if np.array_equal(self.img[y, x], [255, 255, 255]):
                    tmp = Node(x, y, node, iteration + self.heuristic(x, y, self.dst))
                    # Check if node exists in queue, if it does set minimum possible weight
                    if not queue.decreaseWeight(tmp):
                        # Else add node to queue
                        queue.insert(tmp)
            visited.add(node)

        # ===== Retrace path =====
        while (node.x, node.y) != (-1, -1):
            self.img[node.y, node.x] = [0, 0, 255]
            node = node.via

        # ===== Write Image =====
        return self.write()

    def mitmDijakstra(self):
        """
            Apply Dijakstra from src and dst
            If an intersection is found,trace the path
        """
        # ===== Init =====
        src = Node(*self.src, (-1, -1), 0)
        dst = Node(*self.dst, (-2, -2), 0)
        srcData = {"queue": PriorityQueue(), "visited": set(), "dst": self.dst}
        dstData = {"queue": PriorityQueue(), "visited": set(), "dst": self.src}
        srcData["queue"].insert(src)
        dstData["queue"].insert(dst)
        common = set()
        iteration = 0

        # ===== Algorithm =====
        while len(common) == 0:
            iteration += 1
            for ind, data in enumerate([srcData, dstData]):
                node = data["queue"].extract_min()
                if node in data["visited"]:
                    continue
                if self.visualization:
                    self.img[node.y, node.x] = [255, 0, 0] if ind else [0, 255, 0]
                if iteration % self.visualization_speed == 0:
                    self.showFrame(node)
                for x, y in self.getAdjacent(node):
                    # Check if (x,y) is white node
                    if np.array_equal(self.img[y, x], [255, 255, 255]):
                        tmp = Node(x, y, node, iteration)
                        # Check if node exists in queue, if it does set minimum possible weight
                        if not data["queue"].decreaseWeight(tmp):
                            # Else add node to queue
                            data["queue"].insert(tmp)
                data["visited"].add(node)
            common = srcData["visited"].intersection(dstData["visited"])
        common = list(common)[0]

        # ===== Backtrack =====
        nodeSrc = srcData["visited"].pop()
        nodeDst = dstData["visited"].pop()
        while nodeDst != common:
            nodeDst = dstData["visited"].pop()
        while nodeSrc != common:
            nodeSrc = srcData["visited"].pop()

        # ===== Retrace path =====
        while (nodeSrc.x, nodeSrc.y) != self.src:
            self.img[nodeSrc.y, nodeSrc.x] = [0, 0, 255]
            nodeSrc = nodeSrc.via
        self.img[nodeSrc.y, nodeSrc.x] = [0, 0, 255]

        while (nodeDst.x, nodeDst.y) != self.dst:
            self.img[nodeDst.y, nodeDst.x] = [0, 0, 255]
            nodeDst = nodeDst.via
        self.img[nodeDst.y, nodeDst.x] = [0, 0, 255]

        # ===== Write Changes =====
        return self.write()

    def mitmAstar(self):
        """
            Apply Astar from src and dst
            If an intersection is found,trace the path
        """
        # ===== Init =====
        src = Node(*self.src, (-1, -1), 0)
        dst = Node(*self.dst, (-2, -2), 0)
        srcData = {"queue": PriorityQueue(), "visited": set(), "dst": self.dst}
        dstData = {"queue": PriorityQueue(), "visited": set(), "dst": self.src}
        srcData["queue"].insert(src)
        dstData["queue"].insert(dst)
        common = set()
        iteration = 0

        # ===== Algorithm =====
        while len(common) == 0:
            iteration += 1
            for ind, data in enumerate([srcData, dstData]):
                node = data["queue"].extract_min()
                if node in data["visited"]:
                    continue
                if self.visualization:
                    self.img[node.y, node.x] = [255, 0, 0] if ind else [0, 255, 0]
                if iteration % self.visualization_speed == 0:
                    self.showFrame(node)
                for x, y in self.getAdjacent(node):
                    # Check if (x,y) is white node
                    if np.array_equal(self.img[y, x], [255, 255, 255]):
                        tmp = Node(
                            x, y, node, iteration + self.heuristic(x, y, data["dst"])
                        )
                        # Check if node exists in queue, if it does set minimum possible weight
                        if not data["queue"].decreaseWeight(tmp):
                            # Else add node to queue
                            data["queue"].insert(tmp)
                data["visited"].add(node)
            common = srcData["visited"].intersection(dstData["visited"])
        common = list(common)[0]

        # ===== Backtrack =====
        nodeSrc = srcData["visited"].pop()
        nodeDst = dstData["visited"].pop()
        while nodeDst != common:
            nodeDst = dstData["visited"].pop()
        while nodeSrc != common:
            nodeSrc = srcData["visited"].pop()

        # ===== Retrace path =====
        while (nodeSrc.x, nodeSrc.y) != self.src:
            self.img[nodeSrc.y, nodeSrc.x] = [0, 0, 255]
            nodeSrc = nodeSrc.via
        self.img[nodeSrc.y, nodeSrc.x] = [0, 0, 255]

        while (nodeDst.x, nodeDst.y) != self.dst:
            self.img[nodeDst.y, nodeDst.x] = [0, 0, 255]
            nodeDst = nodeDst.via
        self.img[nodeDst.y, nodeDst.x] = [0, 0, 255]

        # ===== Write Changes =====
        return self.write()

    def write(self):
        """
            - Write Image to solved.png
            - Before writeing postProcess it
            - Scale the image to previous size
        """
        cv2.imwrite("solved.png", postProcess(self.img, self.size))
        return self.img


def display(img, delay=1):
    cv2.imshow("Frame", img)
    k = cv2.waitKey(delay)
    if k == 27:
        cv2.destroyAllWindows()
        exit(0)


if __name__ == "__main__":
    print("Run main.py")
