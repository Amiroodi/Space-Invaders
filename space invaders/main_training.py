#main_training.py
import os
import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from a2c_worker import A2CNetwork, A2CWorker
from prioritized_replay_buffer import PrioritizedReplayBuffer
from torch.multiprocessing import Lock

# Worker function for multiprocessing
def worker_process(global_network_state, input_dim, action_space, worker_id, exploration_steps, update_frequency, py_Queue, evt, epsilon):
    worker_device = torch.device("cpu")
    worker = A2CWorker(global_network_state, input_dim, action_space, worker_id, exploration_steps, epsilon)
    worker.run_step(update_frequency)
    experience = worker.get_experience()

    py_Queue.put(experience)  # Push experience data into the queue
    evt.wait()  # Wait for the signal from the main process to proceed


if __name__ == "__main__":
    # Hyperparameters
    input_dim = (4, 84, 84)
    action_space = 6
    learning_rate = 0.00025  # Reduce the learning rate for stability
    total_steps = 100000  # Total steps to train
    update_frequency =  200  # Number of steps before updating the global model
    exploration_steps = 2000  # Number of global steps where only random actions are taken
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay = 100000  # Steps until epsilon reaches epsilon_end
    num_workers = max(1, mp.cpu_count() - 4)  # Number of workers
    batch_size = 64
    replay_buffer_capacity = 40000  # Capacity of the replay buffer
    alpha = 0.6  # Alpha for prioritization
    beta_start = 0.4  # Initial value of beta for importance sampling
    beta_increment_per_step = 0.001  # How much to increment beta after each step

    # Initialize TensorBoard for logging
    writer = SummaryWriter(log_dir=os.path.join(os.getcwd(), "runs/A2C"))

    # Initialize shared replay buffer
    replay_buffer = PrioritizedReplayBuffer(replay_buffer_capacity, alpha)
    buffer_lock = Lock()

    # Model saving parameters
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_filename = "a2c_model_checkpoint.pth"

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize global network
    global_network = A2CNetwork(input_dim[0], action_space).to(device)
    global_network.share_memory()

    # Define the optimizer
    optimizer = optim.Adam(global_network.parameters(), lr=learning_rate)

    # Load checkpoint if exists
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        global_network.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    #Start worker
    mp.set_start_method('spawn', force=True)  # Ensure proper multiprocessing setup
    py_Queue = mp.Queue(maxsize=1000)
    evt = mp.Event()
    global_step_count = 0
    beta = beta_start 

    while global_step_count < total_steps:
        epsilon_start = max(epsilon_end, epsilon_start - ((epsilon_start - epsilon_end) / epsilon_decay))
        processes = []
        
        # Launch worker processes
        for worker_id in range(num_workers):
            p = mp.Process(target=worker_process, args=(global_network.state_dict(), input_dim, action_space, worker_id, exploration_steps, update_frequency, py_Queue, evt, epsilon_start))
            p.start()
            processes.append(p)

        # Collect experiences from the workers
        Queue = []
        for _ in range(num_workers):
            worker_data = py_Queue.get()  # Get data from the queue
            if worker_data:
                Queue.append(worker_data)

        evt.set()  # Signal workers to continue after gathering experience data
        
        # Ensure all workers have finished
        for p in processes:
            p.join(timeout=15)  # Wait for 5 seconds for the worker to finish
            if p.is_alive():
                print(f"Worker {p} is taking too long to finish.")

        # Combine experiences for training
        all_states, all_actions, all_rewards, all_next_states, all_dones = [], [], [], [], []
        episode_rewards = []  # List to track rewards of completed episodes in the batch
        
        for worker_data in Queue:
            all_states.extend(worker_data[0])
            all_actions.extend(worker_data[1])
            all_rewards.extend(worker_data[2])
            all_next_states.extend(worker_data[3])
            all_dones.extend(worker_data[4])
        
        episode_rewards = np.mean(worker_data[5])
        writer.add_scalar("Reward/Average_Reward", episode_rewards , global_step_count)
        print(f"Global Step: {global_step_count}, Average Reward: {episode_rewards }")

        # Store the collected experiences into the replay buffer
        for i in range(len(all_states)):
            # You might want to calculate TD error here, but using a placeholder for now
            td_error = np.abs(all_rewards[i])  # Placeholder for the actual TD error
            with buffer_lock:
                replay_buffer.add(all_states[i], all_actions[i], all_rewards[i], all_next_states[i], all_dones[i], td_error)

        # Training step if we have enough samples
        if global_step_count * num_workers  > batch_size:
            beta = min(1.0, beta + beta_increment_per_step)  # Increment beta for importance sampling
            states, actions, rewards, next_states, dones, indices, is_weights = replay_buffer.sample(batch_size, beta)

            states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
            actions = torch.tensor(np.array(actions), dtype=torch.int64).to(device)
            rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(device)
            next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
            dones = torch.tensor(np.array(dones), dtype=torch.float32).to(device)

            q_values, values = global_network(states)
            _, next_values = global_network(next_states)

            target_values = rewards + 0.99 * next_values.max(1)[0] * (1 - dones)

            # Clamping action probabilities
            action_probs = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            action_probs = torch.clamp(action_probs, 1e-6, 1.0)  # Prevent log(0)
            
            policy_loss = -torch.mean((target_values.detach() - values.squeeze()) * torch.log(action_probs))
            value_loss = F.mse_loss(values.squeeze(), target_values.detach())

            total_loss = policy_loss + value_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(global_network.parameters(), max_norm=1.0)

            optimizer.step()
            
            # Update priorities in the replay buffer based on the TD error
            td_errors = (target_values - values.squeeze()).detach().cpu().numpy()
            replay_buffer.update_priorities(indices, td_errors)
            
            # Save model periodically
            if global_step_count % update_frequency == 0:
                torch.save({
                    'model_state_dict': global_network.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, checkpoint_path)
                print(f"Checkpoint saved at step {global_step_count}")

            # Log training
            writer.add_scalar("Loss/Total_Loss", total_loss.item(), global_step_count)
            print(f"Global Step: {global_step_count}, Loss: {total_loss.item()}")

        global_step_count += update_frequency

    writer.close()
    print("Training complete!")