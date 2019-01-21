# DQN_to_drive_in_TORCS

The input of the DQN_angent is a front camera's images
The outputs are three actions:
	steer: 方向, 取值范围 [-1,1]
	accel： 油门，取值范围 [0,1]
	brake: 刹车，取值范围 [0,1]

	use the activation function: tf.nn.tanh()

reference:

https://github.com/lucianzhong/DQN_to_play_Flappy_Bird/blob/master/DQN_angent.py


How to run?
sudo python DQN_TORCS.py

The files:
gym_torcs.py is the sensor configuration file for TORCS




The pseudo-code for the DQN:

Initialize replay memory D to size N
Initialize action-value function Q with random weights
for episode = 1, M do
    Initialize state s_1
    for t = 1, T do
        With probability ϵ select random action a_t
        otherwise select a_t=max_a  Q(s_t,a; θ_i)
        Execute action a_t in emulator and observe r_t and s_(t+1)
        Store transition (s_t,a_t,r_t,s_(t+1)) in D
        Sample a minibatch of transitions (s_j,a_j,r_j,s_(j+1)) from D
        Set y_j:=
            r_j for terminal s_(j+1)
            r_j+γ*max_(a^' )  Q(s_(j+1),a'; θ_i) for non-terminal s_(j+1)
        Perform a gradient step on (y_j-Q(s_j,a_j; θ_i))^2 with respect to θ
    end for
end for



Still have bugs due to : 2019-01-21 10:44:42.124830: W tensorflow/core/framework/allocator.cc:113] Allocation of 26214400 exceeds 10% of system memory.