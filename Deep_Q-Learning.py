import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from collections import deque
import cv2
import gymnasium as gym
import ale_py

print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

gym.register_envs(ale_py)

# 設定環境
env = gym.make(
    "ALE/SpaceInvaders-v5", 
    #render_mode="human",
    #frameskip=5
    )

# 參數設定
state_size = (84, 84, 4)  # 轉換遊戲畫面為 84x84 灰階並堆疊 4 幀
action_size = env.action_space.n
gamma = 0.99  # 折扣因子
learning_rate = 0.00025
batch_size = 32
memory_size = 100000
epsilon = 1.0  # 初始探索率
epsilon_min = 0.1  # 最小探索率
epsilon_decay = 0.995  # 探索衰減率
update_target_every = 1000  # 目標網路更新頻率

# 儲存經驗回放的記憶體
memory = deque(maxlen=memory_size)

# 讀取預訓練權重
load_pre_train = True
model_weight_name = 'dqn_for_space_invaders_ram_v5_v20250314.keras'

# 預處理函數
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # 轉灰階
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)  # 縮放到 84x84
    return resized / 255.0  # 標準化

# 建立 DQN 網路
def build_model():
    model = keras.Sequential([
        keras.layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=state_size),
        keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
        keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(action_size, activation='linear')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model

# 初始化網路
if load_pre_train:
    try:
        model = tf.keras.models.load_model(model_weight_name)
        target_model = tf.keras.models.load_model(model_weight_name)
        print('啟動權重加載 -> model 和 target model 加載完成')

    except:
        model = build_model()
        target_model = build_model()
        target_model.set_weights(model.get_weights())  # 初始時同步
        print('啟動權重加載 -> model 和 target model加載失敗，重新初始化建立新模型')

else:
    model = build_model()
    target_model = build_model()
    target_model.set_weights(model.get_weights())  # 初始時同步
    print('未啟動權重加載 -> model 和 target model初始化完成')

# 記憶回放函數
def replay():
    if len(memory) < batch_size:
        return
    
    minibatch = random.sample(memory, batch_size)
    
    states, actions, rewards, next_states, dones = zip(*minibatch)
    states = np.array(states)
    next_states = np.array(next_states)
    
    target_q = model.predict(states, verbose=0)
    next_q = target_model.predict(next_states, verbose=0)
    
    for i in range(batch_size):
        target_q[i, actions[i]] = rewards[i] if dones[i] else rewards[i] + gamma * np.max(next_q[i])
    
    model.fit(states, target_q, epochs=1, verbose=0)

# 訓練 DQN
episodes = 10000
steps = 0

for episode in range(episodes):
    frame, _ = env.reset()
    state = np.stack([preprocess_frame(frame)] * 4, axis=-1)  # 初始狀態
    total_reward = 0
    done = False
    
    while not done:
        steps += 1
        if np.random.rand() <= epsilon:
            action = env.action_space.sample()  # 隨機探索
        else:
            action = np.argmax(model.predict(np.expand_dims(state, axis=0), verbose=0))  # 選擇最優動作
        
        next_frame, reward, done, _, _ = env.step(action)
        next_frame = preprocess_frame(next_frame)
        next_state = np.append(state[:, :, 1:], np.expand_dims(next_frame, axis=-1), axis=-1)  # 更新狀態
        
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward
        
        replay()  # 訓練
        
        if steps % update_target_every == 0:
            target_model.set_weights(model.get_weights())  # 更新目標網路
    
    epsilon = max(epsilon_min, epsilon * epsilon_decay)  # 探索率遞減
    print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}")

    if episode % 20 == 0:
        print(f'第{episode}場結束，儲存模型架構與參數')
        model.save(model_weight_name)

print(f'遊戲結束，儲存模型架構與參數')
model.save(model_weight_name)

env.close()
