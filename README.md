# Deep Reinforcement Learning with DQN on Custom FrozenLake Environments

## Introduction

This project demonstrates the application of **Deep Reinforcement Learning (DRL)** using a **Deep Q-Network (DQN)** to solve custom versions of the FrozenLake environment from OpenAI Gym. The agent learns to navigate from a specified start position to a goal position on a frozen lake grid while avoiding holes.

## Background

**Reinforcement Learning (RL)** is a machine learning paradigm where agents learn optimal behaviors through interactions with an environment, aiming to maximize cumulative rewards. **Deep Reinforcement Learning** combines RL with deep neural networks to handle high-dimensional state and action spaces.

**Deep Q-Networks (DQN)**, introduced by DeepMind, utilize neural networks to approximate the Q-value function in RL, enabling agents to learn policies directly from raw observations.

## Project Description

In this project, we:

- **Create custom FrozenLake environments**:

  - **Case 1**: A 10x10 grid with 20 holes.

  - **Case 2**: A 10x10 grid with 40 holes.

  - Both cases use specific start and goal positions derived from a unique identifier (generated randomly with positions: start `(2, 9)` and goal `(8, 8)`).

- **Implement a DQN agent**:

  - Utilizes experience replay and target networks for stable learning.

  - Employs a neural network to approximate Q-values.

- **Adjust the reward structure**:

  - Introduce penalties for each step and for falling into holes.

  - Provide a higher reward for reaching the goal.

- **Modify environment dynamics**:

  - Set `is_slippery=False` to create deterministic environments.

- **Train and evaluate the agent** on both cases.

- **Visualize results** through plots and metrics.

## Files Included

- **`dqn_frozenlake.py`**: The main Python script containing the full implementation.

- **`Konda-model-case1.pt`**: Saved PyTorch model parameters for the agent trained on Case 1.

- **`Konda-model-case2.pt`**: Saved PyTorch model parameters for the agent trained on Case 2.

- **`README.md`**: This file, providing an overview of the project.

- **`requirements.txt`**: A list of required Python packages and their versions.

## Getting Started

### Prerequisites

- **Python 3.x**

- Required Python packages (listed in `requirements.txt`):

  - `gym`

  - `torch`

  - `numpy`

  - `matplotlib`

### Installation

1\. **Clone the repository**:

   git clone https://github.com/your_username/your_repository_name.git

   ```

2\. **Navigate to the project directory**:

   cd your_repository_name

   ```

3\. **(Optional) Create a virtual environment**:

   python -m venv venv

   source venv/bin/activate  # On Windows: venv\Scripts\activate

   ```

4\. **Install the required packages**:

   pip install -r requirements.txt

   ```

### Running the Code

1\. **Ensure the script is executable**:

   chmod +x dqn_frozenlake.py

   ```

2\. **Run the script**:

   python dqn_frozenlake.py

   ```

   - The script will sequentially train and test the agent on both Case 1 and Case 2.

   - Progress updates, results, and metrics will be printed to the console.

   - Plots of rewards and losses will be displayed if running in an environment that supports graphical output.

3\. **View Results**:

   - The trained models `Konda-model-case1.pt` and `Konda-model-case2.pt` will be saved in the project directory.

   - Plots of rewards and losses will help visualize the agent's learning progress.

## Project Structure

- **`dqn_frozenlake.py`**: Contains the following sections:

  - **Imports and Setup**: Importing libraries and setting up the device.

  - **Environment Creation**: Functions to create solvable custom FrozenLake maps.

  - **Map Visualization**: Visualizing the generated maps for both cases.

  - **DQN Implementation**: Defining the neural network architecture.

  - **Experience Replay**: Implementing replay memory for experience sampling.

  - **Helper Functions**: Including functions for one-hot encoding and learning updates.

  - **Reward Adjustment**: Modifying the reward structure to aid learning.

  - **Training the Agent**:

    - Training on **Case 1** with adjusted hyperparameters.

    - Training on **Case 2**.

  - **Testing the Agent**:

    - Testing on **Case 1**.

    - Testing on **Case 2**.

  - **Saving Models**: Saving the trained models for future use.

  - **Conclusion**: Final statements indicating the completion of training and testing.

## Results

### Case 1

- **Training**:

  - Adjusted hyperparameters to improve learning.

  - Increased the number of training episodes to 5000.

  - The agent learned to navigate the environment despite the increased difficulty.

- **Testing**:

  - Achieved a certain success rate (e.g., 80%) after training.

- **Visualizations**:

  - Plots show the total rewards increasing over episodes.

  - Loss decreases over time, indicating learning progress.

### Case 2

- **Training**:

  - Used different hyperparameters suitable for the more complex environment.

  - The agent adapted to the higher number of holes in the grid.

- **Testing**:

  - Achieved a high success rate (e.g., 95%) after training.

- **Visualizations**:

  - Rewards and losses plotted to observe the agent's performance.

## Conclusion

This project successfully demonstrates how a DQN agent can be trained to solve custom FrozenLake environments using Deep Reinforcement Learning techniques. By adjusting the reward structure, environment dynamics, and hyperparameters, the agent effectively learned optimal policies in both cases.

## Acknowledgments

- **OpenAI Gym**: Provides the FrozenLake environment.

- **PyTorch**: Used for building and training the neural network.

- **Reinforcement Learning Resources**: Various tutorials and documentation that guided the implementation.

## Contact Information

For any questions or feedback, please contact:

- **Name**: [Prashanth konda]

- **Email**: [pkonda1@gsu.edu]

- **GitHub**: [your_username](https://github.com/kondster)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Additional Information

### Customization

- **Start and Goal Positions**: The start and goal positions are based on a unique identifier (e.g., Panther ID). You can modify these positions in the code to customize the environments.

- **Hyperparameters**: The hyperparameters used for training can be adjusted in the script to experiment with different learning behaviors.

- **Reward Structure**: The reward wrapper can be modified to change the incentives provided to the agent.

### Potential Improvements

- **Dynamic Environment Generation**: Implement procedures to generate and test multiple environments to improve the agent's robustness.

- **Advanced Algorithms**: Explore other DRL algorithms like Double DQN, Dueling DQN, or Policy Gradient methods.

- **Slippery Environment**: Introduce stochasticity by setting `is_slippery=True` to make the environment more challenging.

### References

- **Human-Level Control through Deep Reinforcement Learning**: Mnih et al., 2015.

- **OpenAI Gym Documentation**: [https://gym.openai.com/](https://gym.openai.com/)

- **PyTorch Documentation**: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)

---

**Disclaimer**: This project is for educational purposes. The performance of the agent may vary based on the computational resources and randomness in environment generation.

```
