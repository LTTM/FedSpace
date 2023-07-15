class ProtoQueue:

    def __init__(self, n_classes, max_length):
        self.n_classes = n_classes
        self.queue = {i: [] for i in range(n_classes)}
        self.max_length = max_length
        self.global_proto = {i: 0 for i in range(n_classes)}

    def insert(self, local_proto, local_radius, num_samples):
        for class_id in local_proto.keys():
            self.queue[class_id].append((local_proto[class_id], local_radius, num_samples[class_id]))

            while len(self.queue[class_id]) > self.max_length:
                self.queue[class_id].pop(0)

    def get_num_samples(self, q_id,  class_id):
        return self.queue[q_id][2][class_id]

    def compute_mean(self):
        for class_id in range(self.n_classes):
            if len(self.queue[class_id]) > 1:
                sum = 0
                ws = 0
                for item in self.queue[class_id]:
                    ws += item[2]
                    sum += item[0] * item[2]

                self.global_proto[class_id] = sum / ws

        return self.global_proto
