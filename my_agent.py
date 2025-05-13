from random import random
import numpy as np
import pygame
from pytorch_mlp import MLPRegression
import argparse
from console import FlappyBirdEnv

STUDENT_ID = 'a1869387'
DEGREE = 'UG'  # or 'PG'


class MyAgent:
    def __init__(self, show_screen=False, load_model_path=None, mode=None):
        # do not modify these
        self.show_screen = show_screen
        if mode is None:
            self.mode = 'train'  # mode is either 'train' or 'eval', we will set the mode of your agent to eval mode
        else:
            self.mode = mode

        # modify these
        self.storage = []  # D in the algorithm — use list as a replay buffer

        # Assume input is 8 (game state features), output is 2 (flap or do nothing)
        self.network = MLPRegression(input_dim=8, output_dim=2, learning_rate=1e-3)

        self.network2 = MLPRegression(input_dim=8, output_dim=2, learning_rate=1e-3)

        # Initialise Q_f's parameters by Q's
        MyAgent.update_network_model(net_to_update=self.network2, net_as_source=self.network)

        self.epsilon = 1.0  # exploration probability
        self.n = 32  # batch size for training
        self.discount_factor = 0.99  # γ in the algorithm

        # do not modify this
        if load_model_path:
            self.load_model(load_model_path)

    def choose_action(self, st, A):
        # Build state representation φ_t from current state s_t
        phi_t = self.build_state(st)

        if self.mode == 'train':
            if random.random() < self.epsilon:
                # Exploration: choose random action
                at = random.choice(A)
            else:
                # Exploitation: choose action with highest Q-value
                q_values = self.network.predict(phi_t)
                at = A[np.argmax(q_values)]

            # Store partial transition (phi_t, a_t, r_t=None, q_t+1=None)
            self.storage.append((phi_t, at, None, None))

        elif self.mode == 'eval':
            # Always choose the best action
            q_values = self.network.predict(phi_t)
            at = A[np.argmax(q_values)]

        return at

    def receive_after_action_observation(self, st1, A):
        if self.mode != 'train':
            return  # Do nothing in evaluation mode

        # Step 1: Build φ_{t+1}
        phi_t1 = self.build_state(st1)

        # Step 2: Compute reward
        rt = self.reward(st1, phi_t1)

        # Step 3: Compute q_{t+1} using Q_f (target network)
        is_terminal = self.is_terminal(st1)
        if is_terminal:
            qt1 = 0
        else:
            q_values_t1 = self.network2.predict(phi_t1)
            qt1 = np.max(q_values_t1)

        # Step 4: Update last transition in D
        phi_t, at, _, _ = self.storage[-1]
        self.storage[-1] = (phi_t, at, rt, qt1)

        # Step 5: Sample minibatch and train
        if len(self.storage) < self.n:
            return  # Not enough samples to train

        minibatch = random.sample(self.storage, self.n)
        X, Y, W = [], [], []

        for phi_j, a_j, r_j, q_j1 in minibatch:
            # w_j = one-hot vector for action a_j
            w_j = np.zeros(len(A))
            w_j[a_j] = 1

            # y_j = r_j + γ * q_{j+1}
            y_j = r_j + self.discount_factor * q_j1

            # Add to minibatch
            X.append(phi_j)
            Y.append(y_j)
            W.append(w_j)

        # Step 6: Train network Q on one batch
        self.network.fit(X, Y, W)

        # Step 7: Optional epsilon decay
        if self.epsilon > 0.01:
            self.epsilon *= 0.995

    def save_model(self, path: str = 'my_model.ckpt'):
        """
        Save the MLP model. Unless you decide to implement the MLP model yourself, do not modify this function.

        Args:
            path: the full path to save the model weights, ending with the file name and extension

        Returns:

        """
        self.network.save_model(path=path)

    def load_model(self, path: str = 'my_model.ckpt'):
        """
        Load the MLP model weights.  Unless you decide to implement the MLP model yourself, do not modify this function.
        Args:
            path: the full path to load the model weights, ending with the file name and extension

        Returns:

        """
        self.network.load_model(path=path)

    @staticmethod
    def update_network_model(net_to_update: MLPRegression, net_as_source: MLPRegression):
        """
        Update one MLP model's model parameter by the parameter of another MLP model.
        Args:
            net_to_update: the MLP to be updated
            net_as_source: the MLP to supply the model parameters

        Returns:
            None
        """
        net_to_update.load_state_dict(net_as_source.state_dict())


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--level', type=int, default=1)
    args = parser.parse_args()

    # Create environment and agent
    env = FlappyBirdEnv(config_file_path='config.yml', show_screen=True, level=args.level, game_length=10)
    agent = MyAgent(show_screen=True)

    episodes = 10000
    best_score = float('-inf')

    for episode in range(episodes):
        env.play(player=agent)

        print(f"Episode {episode+1} — Score: {env.score}, Mileage: {env.mileage}")

        # Save best model based on score
        if env.score > best_score:
            best_score = env.score
            agent.save_model(path='my_model.ckpt')
            print("✔️ New best model saved.")

        # Clear memory every 5 episodes (optional)
        if (episode + 1) % 5 == 0:
            agent.storage.clear()

        # Update target Q-network (Q_f) every 10 episodes
        if (episode + 1) % 10 == 0:
            MyAgent.update_network_model(net_to_update=agent.network2, net_as_source=agent.network)
            print("🔄 Q_f updated.")

    # Evaluation mode
    env2 = FlappyBirdEnv(config_file_path='config.yml', show_screen=False, level=args.level)
    agent2 = MyAgent(show_screen=False, load_model_path='my_model.ckpt', mode='eval')

    episodes = 10
    scores = []

    for episode in range(episodes):
        env2.play(player=agent2)
        scores.append(env2.score)

    print("🏁 Evaluation Results:")
    print("Max score:", np.max(scores))
    print("Average score:", np.mean(scores))
