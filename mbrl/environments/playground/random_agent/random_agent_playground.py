import numpy as np
from gym.utils import seeding


class DirectedRandomAgentwTime:
    def __init__(self, action_dim, nObjects, max_persistency=50, min_persistency=10, epsilon=0.0, seed=None):
        self.action_dim = action_dim
        self.nObjects = nObjects
        self.chosen_object = nObjects  # Placeholder value
        self.max_persistency = max_persistency
        self.min_persistency = min_persistency
        self.patience = 40
        # Take random action with probability epsilon or perform directed action towards the randomly chosen object
        self.epsilon = epsilon
        self.seed(seed)
        # self.reset()

    def reset(self, prev_object=None):
        # At each point a random object will be chosen randomly
        if prev_object is None:
            prev_object = self.nObjects
        while self.chosen_object == prev_object:
            self.chosen_object = np.random.choice(self.nObjects)
        self.chosen_object_state = None
        self.action = np.random.uniform(-1, 1, self.action_dim)

        # Interaction length for object chosen randomly
        self.persistency = np.random.randint(self.min_persistency, self.max_persistency)
        self.offset = None

        # Resetting counters
        self.counter = 0
        self.patience_counter = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def sample(self, state_dict):
        if (self.counter < self.persistency) and (self.patience_counter < self.patience):
            agent_state = state_dict["agent"][:, :2]
            current_obj_state = state_dict["objects_dyn"][self.chosen_object][:, :2]
            if self.chosen_object_state is None:
                self.chosen_object_state = current_obj_state
            if self.offset is None:
                self.offset = np.random.uniform(-0.15, 0.15)
            diff = (self.chosen_object_state + self.offset) - agent_state

            if np.all(np.absolute(diff) < 0.01) and np.any(self.chosen_object_state != current_obj_state):
                # Object has been moved (object state has changed and agent is at the object, the latter is checked to
                #  make sure the dynamic objects have actually been moved by the agent (e.g. ball)

                # Repeat same directed action as applied for the first move with just a bit of randomness
                action = (self.action + np.random.uniform(-0.2, 0.2, self.action_dim)) * np.random.uniform(0.8, 1)
                self.counter += 1
                self.patience_counter = 0  # Resetting the patience counter as the object has been moved
            elif (
                self.counter == 0
            ):  # If the target object hasn't been reached or moved yet, take action in that direction
                if np.random.uniform(0, 1) > self.epsilon:
                    self.action = diff / np.linalg.norm(diff)
                    action = self.action * np.random.uniform(0, 1)
                else:
                    action = np.random.uniform(-1, 1, self.action_dim)
                if np.all(np.absolute(diff) < 0.25):
                    # This part is added for when an object has been moved to the wall, so no change in object state and counter isn't updated
                    self.patience_counter += 1
            else:  # If the counter has been started, i.e. object moved once, but is now far away, reset the chosen_object state
                # This is usually the case when the agent moves together with the object and pushes it around
                self.counter += 1
                self.chosen_object_state = current_obj_state
                diff = (self.chosen_object_state + self.offset) - agent_state
                self.action = diff / np.linalg.norm(diff)
                action = self.action
                # if self.counter > 35:
                #     action = (diff * np.random.uniform(0, 0.8, self.action_dim)) + np.random.uniform(-0.5, 0.5, self.action_dim)
        else:
            self.reset(prev_object=self.chosen_object)
            self.chosen_object_state = state_dict["objects_dyn"][self.chosen_object][:, :2]
            action = self.action
        return np.squeeze(np.clip(action, -1, 1))


class DirectedRandomAgentwTime_offset_static:
    def __init__(self, action_dim, nObjects, max_persistency=50, min_persistency=10, epsilon=0.0, seed=None):
        self.action_dim = action_dim
        self.nObjects = nObjects
        self.chosen_object = nObjects  # Placeholder value
        self.max_persistency = max_persistency
        self.min_persistency = min_persistency
        self.patience = 40
        # Take random action with probability epsilon or perform directed action towards the randomly chosen object
        self.epsilon = epsilon
        self.seed(seed)
        # self.reset()

    def reset(self, prev_object=None):
        # At each point a random object will be chosen randomly
        if prev_object is None or self.nObjects == 1:
            prev_object = self.nObjects
        while self.chosen_object == prev_object:
            self.chosen_object = np.random.choice(self.nObjects)
        self.chosen_object_state = None
        self.action = np.random.uniform(-1, 1, self.action_dim)

        # Interaction length for object chosen randomly
        self.persistency = np.random.randint(self.min_persistency, self.max_persistency)
        self.offset = None
        # Resetting counters
        self.counter = 0
        self.patience_counter = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def sample(self, state_dict):
        if (self.counter < self.persistency) and (self.patience_counter < self.patience):
            agent_state = state_dict["agent"][:, :2]
            current_obj_state = state_dict["objects_dyn"][self.chosen_object][:, :2]
            if self.chosen_object_state is None:
                self.chosen_object_state = current_obj_state
            if self.offset is None:
                self.offset = np.random.uniform(-0.15, 0.15)
            diff = (self.chosen_object_state + self.offset) - agent_state
            # print('Self chosen object: ', self.chosen_object, self.action)

            if np.all(np.absolute(diff) < 0.01) and np.any(self.chosen_object_state != current_obj_state):
                # Object has been moved (object state has changed and agent is at the object, the latter is checked to
                #  make sure the dynamic objects have actually been moved by the agent (e.g. ball)

                # Repeat same directed action as applied for the first move
                # action = self.action * np.random.uniform(0.4, 0.9)

                action = (self.action + np.random.uniform(-0.2, 0.2, self.action_dim)) * np.random.uniform(0.8, 1)
                self.counter += 1
                self.patience_counter = 0  # Resetting the patience counter as the object has been moved
            elif (
                self.counter == 0
            ):  # If the target object hasn't been reached or moved yet, take action in that direction
                if np.random.uniform(0, 1) > self.epsilon:
                    self.action = diff / np.linalg.norm(diff)
                    action = self.action * np.random.uniform(0, 1)
                else:
                    action = np.random.uniform(-1, 1, self.action_dim)
                if np.all(np.absolute(diff) < 0.25):
                    # This part is added for when an object has been moved to the wall, so no change in object state and counter isn't updated
                    self.patience_counter += 1
            else:  # I the counter has been started, i.e. object moved once, but is now far away, reset the chosen_object state
                self.counter += 1
                # self.chosen_object_state = current_obj_state
                # diff = ((self.chosen_object_state+self.offset) - agent_state)
                # self.action = diff / np.linalg.norm(diff)
                # action = self.action
                action = (self.action + np.random.uniform(-0.2, 0.2, self.action_dim)) * np.random.uniform(0.8, 1)
                # if self.counter > 35:
                #     action = (diff * np.random.uniform(0, 0.8, self.action_dim)) + np.random.uniform(-0.5, 0.5, self.action_dim)
        else:
            self.reset(prev_object=self.chosen_object)
            self.chosen_object_state = state_dict["objects_dyn"][self.chosen_object][:, :2]
            # print('New object: ', self.chosen_object, self.chosen_object_state,
            #       state_dict['objects_dyn'][self.chosen_object][:, :2])
            action = self.action
        return np.squeeze(np.clip(action, -1, 1))
