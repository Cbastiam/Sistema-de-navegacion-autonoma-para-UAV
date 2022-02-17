# --------------------------------------------------------------
# ------------------------- Libraries --------------------------
# --------------------------------------------------------------
import time
import os
import warnings
import math as m
import random
import cv2
import pybullet as pb
import pybullet_data
import numpy as np
import gym
import matplotlib.pyplot as plt
import sim
import imutils 
from gym import spaces
import datetime

with warnings.catch_warnings():    
    warnings.filterwarnings("ignore", category=FutureWarning)
    from stable_baselines.common.env_checker import check_env
    from stable_baselines.deepq.policies import CnnPolicy, MlpPolicy
    from stable_baselines import DQN, results_plotter
    from stable_baselines.common.callbacks import BaseCallback
    from stable_baselines.bench import Monitor
    from stable_baselines.results_plotter import load_results, ts2xy

# --------------------------------------------------------------
# ------------------------- Constants --------------------------
# --------------------------------------------------------------
# Physical Boundaries
MAX_X = 2.5
MIN_X = -2.5
MAX_Y = 5
MIN_Y = -2.5
reward_per_episode = []


from tqdm.auto import tqdm

class ProgressBarCallback(BaseCallback):
    """
    :param pbar: (tqdm.pbar) Progress bar object
    """
    def __init__(self, pbar):
        super(ProgressBarCallback, self).__init__()
        self._pbar = pbar

    def _on_step(self):
        # Update the progress bar:
        self._pbar.n = self.num_timesteps
        self._pbar.update(0)

# this callback uses the 'with' block, allowing for correct initialisation and destruction
class ProgressBarManager(object):
    def __init__(self, total_timesteps): # init object with total timesteps
        self.pbar = None
        self.total_timesteps = total_timesteps
        
    def __enter__(self): # create the progress bar and callback, return the callback
        self.pbar = tqdm(total=self.total_timesteps)
            
        return ProgressBarCallback(self.pbar)

    def __exit__(self, exc_type, exc_val, exc_tb): # close the callback
        self.pbar.n = self.total_timesteps
        self.pbar.update(0)
        self.pbar.close()

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq, log_dir, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model at {} timesteps".format(x[-1]))
                    print("Saving new best model to {}.zip".format(self.save_path))
                  self.model.save(self.save_path)

        return True
    
class quadEnviSim(gym.Env):
     # --------------------------------------------------------------
    # ----------------------- Initialization -----------------------
    # --------------------------------------------------------------

    def __init__(self, max_it=300, max_lost=20):
        super(quadEnviSim, self).__init__()
        self.rewardPerStep = 0
        #Goal Position
        self.objetive = None
        self.prevEuclidianPos = np.inf
        self.nextEuclidianPos = np.inf
        # Initial Position
        self.pos = [0, 0, 0]
        # States
        self.new_state = None
        self.prev_state = None
        self.lost = False
        self.lostCount = 0
        self.obs = [0,0,0,0,0,0]
        # Sim Parameters
        self.max_it = max_it
        self.max_lost = max_lost
        self.client = -1
        self.episode_step = 0
        self.epidose= 0
        self.conectar_cliente()
        self.reset()
        # Action Space
        n_actions = 4
        self.action_space = spaces.Discrete(n_actions)
        # Observation Space
        self.observation_space = spaces.Box(
            low=np.array([-2.5,-2.5,0,0,-2,-2]), high=np.array([2.5,2.5,3,3,2,2]), dtype=np.float32)
        #handlers 
        self.target_hand = 0
        self.Sink = 0
        self.SphericalHand = 0
        #reward

        


    # Configuration of the simulation environment
    def putting_on_stream(self):
        #print('=========================Estoy en puttingString===================================')
        #Obtener handlers
        allDone = False
        while allDone == False:
            returnCode,S=sim.simxGetObjectHandle(self.client,'sphericalVisionDepth_sensor',sim.simx_opmode_blocking)
            Spherical = S
            returnCode,Base=sim.simxGetObjectHandle(self.client,'Quadcopter_base',sim.simx_opmode_blocking)
            Base_H = Base
            returnCode,T=sim.simxGetObjectHandle(self.client,'Quadcopter_target',sim.simx_opmode_blocking)
            Target = T
            rreturnCode,Sink = sim.simxGetObjectHandle(self.client, 'Sink', sim.simx_opmode_blocking)
            #Inicializar el streamingMode en el sensor de vision
            returnCode,resolution,image=sim.simxGetVisionSensorImage(self.client,Spherical,0,sim.simx_opmode_streaming)
            if S!= 0 and Base!=0 and T!=0:
                allDone= True
                #print('ya puse los handlers')
                
        returnCode= -5
        while returnCode !=0:
            returnCode,resolution,image=sim.simxGetVisionSensorImage(self.client,Spherical,0,sim.simx_opmode_streaming)
            time.sleep(1)
        returnCode,resolution,image=sim.simxGetVisionSensorImage(self.client,Spherical,0,sim.simx_opmode_buffer)
        #print('Ya puse el buffer')
        self.target_hand = Target
        self.SphericalHand = Spherical
        self.Sink = Sink
        self.set_random_objetive()
        res, pos_obj = sim.simxGetObjectPosition(self.client, Sink, -1, sim.simx_opmode_blocking)
        self.objetive = pos_obj[0],pos_obj[1],pos_obj[2]
        print('La posicion del objetivo es: ',self.objetive)
        self.pos = sim.simxGetObjectPosition(self.client, self.target_hand, -1, sim.simx_opmode_blocking)
        
        #print('=========================Salgo de puttingString===================================')     
    
    def cargar_escena(self,cliente,scene):
        res = sim.simxLoadScene(cliente,scene,0,sim.simx_opmode_blocking )
        time.sleep(1)
        res = sim.simxStartSimulation(self.client, sim.simx_opmode_oneshot)
        time.sleep(1)
        self.episode_step = 0
        #print('Cargando escena')
        self.putting_on_stream()
                    
    def conectar_cliente(self):
        #print('=========================Estoy en Conectar_a_coppelia===================================')
        # Client numbersim.simxFinish(-1) # just in case, close all opened connections
        sim.simxFinish(-1) # just in case, close all opened connections
        #print('El cliente actual es ', self.client)
        while self.client == -1 :
            
            clientID=sim.simxStart('127.0.0.1',19999,True,True,2000,5) # Conectarse
            if clientID == 0: 
              print("conectado a", 19999)
              self.client = clientID
            else: 
              print("no se pudo conectar")
        #print('=========================Salgo de Conectar_a_coppelia===================================')
    def reset(self):
        print("Reseting")
        self.epidose +=1
        #Resetea la recompensa para el nuevo episodio
        reward_per_episode.append(self.rewardPerStep)
        self.rewardPerStep=0
        #print('=========================Entro a reset===================================')        
        #Detiene la simualcion
        res = sim.simxStopSimulation(self.client, sim.simx_opmode_oneshot)
        #Cierra la escena
        res = sim.simxCloseScene(self.client, sim.simx_opmode_blocking)
        #print('Se cerro la escena')
        #Genera una nueva conexion
        self.set_environment()
        #print('Creando un nuevo ambiente')
        # Update state
        self.new_state = self.update_state()
        # Observation
        self.obs  = self.new_state
        #print('=========================Salgo de reset===================================')



        return np.array(self.obs)
    def set_environment(self):
        #print('=========================Entro a a set_enviroment===================================')

        rutas_scenas = ['C:/Users/kscer/Documents/Universidad/Semestre 8/Tesis/Escenas_Vrep/Train_1_no_obts.ttt']
                   
                        #'C:/Users/kscer/Documents/Universidad/Semestre 8/Tesis/Escenas_Vrep/Train_1_obts.ttt']
                        #'C:/Users/kscer/Documents/Universidad/Semestre 8/Tesis/Escenas_Vrep/Train_2_obts.ttt'
        
        self.scene = random.choice(rutas_scenas)
        #print(' se eligio la escenea ', self.scene)
        #print('Conectando a copelia')
        self.cargar_escena(self.client,self.scene)
        #self.set_random_objetive()
        time.sleep(0.1)
        #print('conectado a coppelia')
        #print('=========================Salgo de set_enviroment===================================')
    # --------------------------------------------------------------
    # ---------------------------- Step ----------------------------
    # --------------------------------------------------------------
    def step(self, action):
        if self.episode_step%1000 == 0:
            print('--------------------------')
        
            
        # Quad movement
        self.move(action)
        # Save previous state
        self.prev_state = self.new_state
        # Updated new state
        self.new_state = self.update_state()
        # Get reward and endEpisode flag
        reward, endEpisode = self.get_reward(
            self.prev_state, self.new_state, self.episode_step)
        # Next step
        self.episode_step += 1
        # Observation
        self.obs = self.new_state
        # Aditional info
        info = {}
        self.rewardPerStep+=reward
        print('---------------------------------------------------------------------------------------------------------')
        print('Episode Step:      ' + str(self.episode_step)+ '\n' +
              'Reward:            '+ str(reward) + '\n' +
              'Acomulated reward: ' + str(self.rewardPerStep)+'\n' +
              'Episode:           ' + str(self.epidose))
        print('---------------------------------------------------------------------------------------------------------')
        #print(self.rewardPerStep)

        return np.array(self.obs, dtype=np.float32) , reward, endEpisode, info
    
    # ---------------------------------------------------------
    # ----------------------- Methods -----------------------
    # ---------------------------------------------------------
    def set_random_objetive(self):

        new_position_objetiveX = np.random.randint(-20,20)
        new_position_objetivey = np.random.randint(-20,20)
        positionForObjetiveX = [new_position_objetiveX/10, new_position_objetivey/10,0]
        #Setting objetive in a random position
        res= sim.simxSetObjectPosition(self.client, self.Sink, -1,positionForObjetiveX, sim.simx_opmode_oneshot)
        returnCode,pos=sim.simxGetObjectPosition(self.client, self.Sink, -1, sim.simx_opmode_blocking)
        print('se establecio la posicion del objetivo en ', pos )

    #  ----------------------- ACTION  ------------------------
    def movement_action(self, action, distance=0.06):
        r,pos = sim.simxGetObjectPosition(self.client, self.target_hand, -1, sim.simx_opmode_blocking)
        #Movimiento hacia adelante
        if action == 0:
            new_pos= [round(pos[0],3),round(pos[1],3)+distance,round(pos[2],3)]
            res= sim.simxSetObjectPosition(self.client, self.target_hand, -1, new_pos , sim.simx_opmode_oneshot)
           
        #Movimiento derecha
        elif action == 1:
            new_pos= [round(pos[0],3)+distance,round(pos[1],3),round(pos[2],3)]
            res= sim.simxSetObjectPosition(self.client, self.target_hand, -1, new_pos, sim.simx_opmode_oneshot)
        #Movimiento atras
        elif action == 2:
            new_pos= [round(pos[0],3),round(pos[1],3)-distance,round(pos[2],3)]
            res= sim.simxSetObjectPosition(self.client, self.target_hand, -1, new_pos, sim.simx_opmode_oneshot)
        #Movimiento izquierda
        elif action == 3:
            new_pos= [round(pos[0],3)-distance,round(pos[1],3),round(pos[2],3)]
            res= sim.simxSetObjectPosition(self.client, self.target_hand, -1, new_pos, sim.simx_opmode_oneshot)
        
    # Low level control
    def move(self, action):
        # Get previous position
        returnCode,pos=sim.simxGetObjectPosition(self.client, self.target_hand, -1, sim.simx_opmode_blocking)
        prevX, prevY, prevW = pos
        # Move
        #print('(Action) quad is moving: ' + str(action))
        self.movement_action(action)
        #time.sleep(0.1)
        # Update position
        
        returnCode,self.pos = sim.simxGetObjectPosition(self.client, self.target_hand, -1, sim.simx_opmode_blocking)
        start_moving = time.time()
        # Low level control to verify movement
        while ((abs(prevX-self.pos[0]) < 0.05) and
                (abs(prevY-self.pos[1]) < 0.05) and ((time.time() - start_moving) < 0.3)):
            # Retry pepper movement
            # print('Spaming Pepper movement')
            self.movement_action(action)
            returnCode,self.pos = sim.simxGetObjectPosition(self.client, self.target_hand, -1, sim.simx_opmode_blocking)
            #time.sleep(0.001)
        #print(self.pos, self.pos[2])
    #  ------------------------ STATE  ------------------------
    def update_state(self):

         # Observation
        res,position= sim.simxGetObjectPosition(self.client, self.target_hand, -1, sim.simx_opmode_blocking) 
        self.obs[0] = position[0]
        self.obs[1] = position[1]
        contador,a,b,c = self.get_close_to_obs()
        self.obs[2] = contador
        contador_estrellos,a,b,c = self.if_get_coll_to_obs()
        self.obs[3] =contador_estrellos
        self.obs[4] = self.objetive[0]
        self.obs[5] = self.objetive[1]
        
        
        return self.obs
    
    
    
    def get_image(self):
        returnCode= -5
        while returnCode !=0:
            returnCode,resolution,image=sim.simxGetVisionSensorImage(self.client,self.SphericalHand,0,sim.simx_opmode_streaming)
            time.sleep(0.01)
        returnCode,resolution,image=sim.simxGetVisionSensorImage(self.client,self.SphericalHand,0,sim.simx_opmode_buffer)

        img = np.array(image, dtype=np.uint8)
    
        img.resize([128, 256,3])
        img = imutils.rotate(img, 180)
        img = cv2.flip(img,1)
        directory = r'C:\Users\kscer\Documents\Universidad\Semestre 8\Tesis\Scripts'
        os.chdir(directory)
        cv2.imwrite('Prueba5.png', img)

        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray_image
    
    def get_close_to_obs(self):
        ancho = 256
        alto = 128
       
        gray_image = self.get_image()
        contador = 0
        umbral = 800
        contador_izquierda = 0
        centinela1 = False
        centinela2 = False
        centinela3 = False
        for fila in range(len(gray_image)):
            for columna in range(len(gray_image[0])):
                if 0<=columna<ancho-156:
                    if gray_image[fila][columna]<=11+8:
                        contador_izquierda+=1
    
        #Examinar parte centro
        contador_centro=0
        for fila in range(len(gray_image)):
            for columna in range(len(gray_image[0])):
                if (ancho-156)<=columna<(ancho-100):
                    if gray_image[fila][columna]<=11+8:
                        contador_centro+=1
        
        #Examinar parte derecha
        contador_derecha=0
        for fila in range(len(gray_image)):
            for columna in range(len(gray_image[0])):
                if (ancho-100)<=columna<ancho:
                   if gray_image[fila][columna]<=11+8:
                        contador_derecha+=1
        if contador_izquierda>umbral:
            contador+=1
            centinela1= True
        if contador_centro >umbral:
            contador+=1
            centinela2= True
        if contador_derecha>umbral:
            centinela3 = True
            contador+=1
        if centinela1 and centinela3 and centinela2== False:
            if contador_derecha> contador_izquierda*2 or contador_izquierda>contador_derecha*2:
                contador = 2
            else : 
                contador=1
        return contador,contador_izquierda,contador_centro,contador_derecha


    #  ------------------------ REWARD  ------------------------
    
    def if_get_coll_to_obs(self):

       
        ancho = 256
        alto= 128

        gray_image = self.get_image()
        #Contador de partes negras 
        contador = 0
        umbral = 800
        contador_izquierda = 0    
        for fila in range(len(gray_image)):
            for columna in range(len(gray_image[0])):
                if 0<=columna<ancho-156:
                    if gray_image[fila][columna]<=9:        
                        contador_izquierda+=1
                        
        #Examinar parte centro
        contador_centro=0
        for fila in range(len(gray_image)):
            for columna in range(len(gray_image[0])):
                if (ancho-156)<=columna<(ancho-100):
                    if gray_image[fila][columna]<=9:
                        contador_centro+=1
        
        #Examinar parte derecha
        contador_derecha=0
        for fila in range(len(gray_image)):
            for columna in range(len(gray_image[0])):
                if (ancho-100)<=columna<ancho:
                   if gray_image[fila][columna]<=9:
                        contador_derecha+=1
        if contador_izquierda>umbral:
            contador+=1
        if contador_centro >umbral:
            contador+=1
        if contador_derecha>umbral:
            contador+=1
    
        
        return contador,contador_izquierda,contador_centro,contador_derecha
    
    
    def get_reward(self, prevState, newState, step):
        # Variables
        reward = 0

        goalReached = False
        outOfBounds = False
        lostForTooLong = False
        collision = False
        maxItReached = False

        # Reward for getting closer to an obstacule
        cant_obj = newState[2]
        if cant_obj== 1:
            reward-= 50
        elif cant_obj==1:
            reward-=50*2
        elif cant_obj == 3:
            reward-= 5*3
        #reward for getting close to the objetive
        a= np.array((newState[0],newState[1]))
        b= np.array((newState[4],newState[5]))
        self.prevEuclidianPos =  self.nextEuclidianPos
        self.nextEuclidianPos = np.linalg.norm(a-b)
        #print('Distancia previa: ', self.prevEuclidianPos, 'distancia actual:', self.nextEuclidianPos)
        if self.nextEuclidianPos<self.prevEuclidianPos:
            #print('Entro a 50')
            reward += 25
        elif self.nextEuclidianPos>self.prevEuclidianPos:
            #print('Entro a 25')
            reward += -35
        # Update position
        # Verify if robot reach goalReached()
        
        if  ((newState[4]-0.15)<=newState[0]<=newState[4]+0.15) and ((newState[5]-0.15)<=newState[1]<=newState[5]+0.15):
        
            goalReached = True
            print('EPISODE TERMINATED: Robot reached goal')
            # self.endCause[0] += 1
            reward += 1000 - step
        
        # Verify if robot out of bounds
        if (newState[0] > MAX_X-0.05) or (newState[1] > MAX_Y-0.05) or (newState[0] < MIN_X+0.05) or (newState[1] < MIN_Y+0.05):
            outOfBounds = True
            reward += -75
            print('EPISODE TERMINATED: Robot out of bounds')
            # self.endCause[1] += 1   
        # Verify if collision with obstacle
        obst_counter = newState[3]
        if obst_counter>0:
            collision = True
            reward += -75
            print('EPISODE TERMINATED: Robot collide with an obstacle')
            
        # Verify if max iterations reached
        if (step > self.max_it):
            maxItReached = True
            reward += -75
            print('EPISODE TERMINATED: Max iterations reached')
            # self.endCause[4] += 1

        # Print final reward
        #if self.episode_step%1000 == 0:
        
        #print('(Reward) r: ' + str(reward/100))
        # End episode condition
        endEpisode = maxItReached or outOfBounds or goalReached or lostForTooLong or collision

        return (reward/100), endEpisode
        
        
    
def main():
    

    
    path_tensorBoard = 'C:/Users/kscer/Documents/Universidad/Semestre 8/Tesis/results/logs/' 
    # Create log dir
    log_dir = "C:/Users/kscer/Documents/Universidad/Semestre 8/Tesis/results/logs/DQN1/"
    # Leaf directory
     
    # Parent Directories
     
    # Create the directory
    # 'ihritik'
    os.makedirs(log_dir, exist_ok=True)
    # Pepper Training Env
    env = quadEnviSim()
    env = Monitor(env, log_dir)
    # Create the callback: check every 1000 steps
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
    
    
    
    #Borrar en la siguiente simualcionnnnnnnnnnnnnnnn
    #cnn_model = DQN.load("C:/Users/kscer/Documents/Universidad/Semestre 8/Tesis/results/logs/DQN1/best_model.zip", env=env, tensorboard_log= path_tensorBoard)
    
    #Model
    cnn_model = DQN(MlpPolicy, env, tensorboard_log= path_tensorBoard, buffer_size=1000, exploration_fraction=0.5,
                    exploration_final_eps=0.025, exploration_initial_eps=1.0,
                    train_freq=1, batch_size=32, double_q=True,
                    learning_starts=1001, target_network_update_freq=250,
                    prioritized_replay=True, verbose=1)
    # Training
    with ProgressBarManager(200000) as progress_callback:
        cnn_model.learn(total_timesteps=200000, callback=[callback,progress_callback], tb_log_name= datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))






if __name__ == "__main__":
    # Main function
    #from stable_baselines.common.env_checker import check_env
    #env = quadEnviSim()
    #check_env(env, warn=True)
    main()    
        
        
 