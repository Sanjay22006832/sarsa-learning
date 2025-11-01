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
<img width="379" height="232" alt="image" src="https://github.com/user-attachments/assets/b9006eb1-6bd9-4030-af32-060007da001c" />

<img width="700" height="591" alt="image" src="https://github.com/user-attachments/assets/729f4126-ae6b-45cb-a16f-b19204137312" />

<img width="1511" height="619" alt="image" src="https://github.com/user-attachments/assets/c593e46c-9f58-4232-803b-47a3546bd608" />
<img width="1494" height="610" alt="image" src="https://github.com/user-attachments/assets/5ab13baf-1e3e-4259-9309-10d8950e54c9" />



## RESULT:
Thus, to implement SARSA learning algorithm is executed successfully.
