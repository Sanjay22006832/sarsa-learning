# SARSA Learning Algorithm


## AIM
To implement SARSA Learning Algorithm.

## PROBLEM STATEMENT
The problem might involve teaching an agent to interact optimally with an environment (e.g., gym-walk), where the agent must learn to choose actions that maximize cumulative rewards using RL algorithms like SARSA and Value Iteration.

## SARSA LEARNING ALGORITHM
1. Initialize the Q-table, learning rate α, discount factor γ, exploration rate ϵ, and the number of episodes.<br>
2. For each episode, start in an initial state s, and choose an action a using the ε-greedy policy.<br>
3. Take action a, observe the reward r and the next state s′ , and choose the next action a′ using the ε-greedy policy.<br>
4. Update the Q-value for the state-action pair (s,a) using the SARSA update rule.<br>
5. Update the current state to s′ and the current action to a′.<br>
6. Repeat steps 3-5 until the episode reaches a terminal state.<br>
7. After each episode, decay the exploration rate 𝜖 and learning rate α, if using decay schedules.<br>
8. Return the Q-table and the learned policy after completing all episodes.<br>

## SARSA LEARNING FUNCTION
### Name: M Sanjay
### Register Number: 212222240090

```python
def sarsa(env,
          gamma=1.0,
          init_alpha=0.5,
          min_alpha=0.01,
          alpha_decay_ratio=0.5,
          init_epsilon=1.0,
          min_epsilon=0.1,
          epsilon_decay_ratio=0.9,
          n_episodes=3000):
    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)

        # Set up decaying schedules for alpha and epsilon
    def decay_schedule(init_value, min_value, decay_ratio, n_episodes):
        decay_episodes = int(n_episodes * decay_ratio)
        values = np.linspace(init_value, min_value, decay_episodes)
        values = np.concatenate((values, np.ones(n_episodes - decay_episodes) * min_value))
        return values

    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    epsilons = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)

    def epsilon_greedy_policy(Q, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        return np.argmax(Q[state])

    for e in range(n_episodes):
        state = env.reset()
        # Compatibility for Gym versions
        if isinstance(state, tuple):
            state = state[0]
        action = epsilon_greedy_policy(Q, state, epsilons[e])
        done = False

        while not done:
            # Modified this line to handle 5 return values from env.step()
            next_state, reward, done, info = env.step(action)


            # Handle Gym version differences
            if isinstance(next_state, tuple):
                next_state = next_state[0]

            next_action = epsilon_greedy_policy(Q, next_state, epsilons[e])
            td_target = reward + gamma * Q[next_state, next_action] * (not done)
            td_error = td_target - Q[state, action]
            Q[state, action] += alphas[e] * td_error

            state, action = next_state, next_action

        Q_track[e] = Q.copy()
        pi = np.argmax(Q, axis=1)
        pi_track.append(pi)

    V = np.max(Q, axis=1)
    pi_array = np.argmax(Q, axis=1)
    pi = lambda s: {s:a for s, a in enumerate(pi_array)}[s]


    return Q, V, pi, Q_track, pi_track
```

## OUTPUT:
<img width="933" height="821" alt="image" src="https://github.com/user-attachments/assets/1a99a4d4-925a-4d03-9e65-71974c4863c9" />
<img width="901" height="257" alt="image" src="https://github.com/user-attachments/assets/b1849519-bded-424f-8728-af3240949dd0" />
<img width="1739" height="652" alt="image" src="https://github.com/user-attachments/assets/72ef0ede-2b7a-484b-9cd7-d7ce48accc30" />
<img width="1733" height="671" alt="image" src="https://github.com/user-attachments/assets/6ced5890-6e1d-4809-a12b-992856bd0b83" />






## RESULT:
Thus, to implement SARSA learning algorithm is executed successfully.
