#PrioritizedReplayBuffer
import numpy as np
import random

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # Tree structure for priorities
        self.write = 0  # Pointer for where to insert next experience

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def add(self, priority):
        idx = self.write + self.capacity - 1  # Determine the leaf node in the tree
        self.update(idx, priority)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0  # Wrap around if the capacity is reached
        return idx  # Return the index of the added experience

    def get_leaf(self, s):
        parent_idx = 0

        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1

            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if s <= self.tree[left_child_idx]:
                    parent_idx = left_child_idx
                else:
                    s -= self.tree[left_child_idx]
                    parent_idx = right_child_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], data_idx

    @property
    def total_priority(self):
        return self.tree[0]


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.epsilon = 1e-6  # Small value to ensure non-zero priority
        self.tree = SumTree(capacity)  # Priority tree
        self.experience_store = []  # Only store experiences here
        self.max_priority = 1.0  # Maximum priority for newly added experiences

    def add(self, state, action, reward, next_state, done, error):
        # Use the TD error for priority calculation
        priority = (np.abs(error) + self.epsilon) ** self.alpha
        experience = (state, action, reward, next_state, done)

        idx = self.tree.add(priority)
        
        # If experience_store is full, replace the experience at the correct index
        if len(self.experience_store) < self.capacity:
            self.experience_store.append(experience)
        else:
            # Overwrite the old experience based on write pointer
            self.experience_store[self.tree.write - 1] = experience
        
        self.max_priority = max(self.max_priority, priority)

    def sample(self, batch_size, beta=0.4):
        batch = []
        idxs = []
        priorities = []
        segment = self.tree.total_priority / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, priority, data_idx = self.tree.get_leaf(s)

            # Fetch experience directly by aligning index
            experience = self.experience_store[data_idx]
            batch.append(experience)
            idxs.append(idx)
            priorities.append(priority)

        # Normalize importance sampling weights
        sampling_probabilities = np.array(priorities) / self.tree.total_priority
        is_weights = np.power(self.tree.capacity * sampling_probabilities, -beta)
        is_weights /= is_weights.max()  # Normalize the weights

        # Unpack batch for training
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), actions, rewards, np.array(next_states), dones, idxs, is_weights

    def update_priorities(self, batch_indices, batch_errors):
        # Simply update priorities in the SumTree without modifying experience_store
        for idx, error in zip(batch_indices, batch_errors):
            priority = (np.abs(error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)