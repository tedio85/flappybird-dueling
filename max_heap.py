import heapq

class MaxHeap(object):
    def __init__(self):
        self.heap = []

    def push(self, sample):
        """
        a sample is a tuple of (priority, s, a, r, t, s2)
        heapq builds a min heap according to the first element of
        the tuple, so we have to take the priority as negative
        to make it a max heap
        """
        tup = sample
        reversed = (-tup[0],) + tup[1:]
        heapq.heappush(self.heap, reversed)

    def pop(self):
        tup = heapq.heappop(self.heap)
        sample = tup[1:]
        priority = -tup[0]
        return sample, priority

    def clear(self):
        del self.heap
        self.heap = []
