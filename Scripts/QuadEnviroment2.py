
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
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.dqn.policies import  MlpPolicy
    from stable_baselines3 import DQN
    #from stable_baselines.common.callbacks import BaseCallback
    #from stable_baselines.bench import Monitor
    #from stable_baselines.results_plotter import load_results, ts2xy
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.results_plotter import load_results, ts2xy
    from stable_baselines3.common.noise import NormalActionNoise
    from stable_baselines3.common.callbacks import BaseCallback
# --------------------------------------------------------------
# ------------------------- Constants --------------------------
# --------------------------------------------------------------
# Physical Boundaries
MAX_X = 2.5
MIN_X = -2.5
MAX_Y = 2.5
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
        self.save_path = os.path.join(log_dir, 'new_best_model')
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
              # Mean training reward over the last 300 episodes
              mean_reward = np.mean(y[-300:])
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
        self.tolUmbral1 = 8
        self.tolUmbral2 = 13
        self.tolUmbral3 = 18
        self.tolUmbral4 = 23
        self.tolUmbral5 = 28
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
        self.obs = [0,0,-1,-1,-1,-1,0,0,0]
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
            low=np.array([-2.5,-2.5,-10,-10,-10,-10,0,-2,-2]), high=np.array([2.5,2.5,2.5,2.5,2.5,2.5,3,2.5,2.5]), dtype=np.float32)
            #[posX,posY,DisObstIzq,DisObstArriba,DisObstDere,estrellos,DisObstAbajo,quadX,quadY]\
        #handlers 
        self.target_hand = 0
        self.Sink = 0
        self.SphericalHand = 0
        self.base = 0


        


    # Configuration of the simulation environment
    def putting_on_stream(self):
        #print('=========================Estoy en puttingString===================================')
        #Obtener handlers
        allDone = False
        while allDone == False:
            returnCode,S=sim.simxGetObjectHandle(self.client,'sphericalVisionDepth_sensor',sim.simx_opmode_blocking)
            Spherical = S
            returnCode,Base=sim.simxGetObjectHandle(self.client,'Quadcopter',sim.simx_opmode_blocking)
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
        self.base = Base_H
        self.set_random_objetive()
        res, pos_obj = sim.simxGetObjectPosition(self.client, Sink, -1, sim.simx_opmode_blocking)
        self.objetive = [pos_obj[0],pos_obj[1],pos_obj[2]]
        #print('La posicion del objetivo es: ',self.objetive)
        res,self.pos = sim.simxGetObjectPosition(self.client, self.target_hand, -1, sim.simx_opmode_blocking)
        #print('La posicion del quad es: ',self.pos)
        #print('=========================Salgo de puttingString===================================')     
    
    def cargar_escena(self,cliente,scene):
        res = sim.simxLoadScene(cliente,scene,0,sim.simx_opmode_blocking )
        time.sleep(1)
        
        self.episode_step = 0
        #print('Cargando escena')
        self.putting_on_stream()
        res = sim.simxStartSimulation(self.client, sim.simx_opmode_oneshot)
                    
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
        print(self.obs)
        #print('=========================Salgo de reset===================================')
        return np.array(self.obs)
    def set_environment(self):
        #print('=========================Entro a a set_enviroment===================================')

        rutas_scenas = [#'C:/Users/kscer/Documents/Universidad/Semestre 8/Tesis/Escenas_Vrep/Train_1_no_obts.ttt']
                        #'C:/Users/kscer/Documents/Universidad/Semestre 8/Tesis/Escenas_Vrep/Train_1_no_obts_validation.ttt']
                        'C:/Users/kscer/Documents/Universidad/Semestre 8/Tesis/Escenas_Vrep/Train_1_obts_MV.ttt']
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
        # Quad movement
        self.move(action)
        # Save previous state
        self.prev_state = self.new_state
        # Updated new state
        self.new_state = self.update_state()
        # Get reward and endEpisode flag
        reward, endEpisode,info = self.get_reward(
            self.prev_state, self.new_state, self.episode_step)
        # Next step
        self.episode_step += 1
        # Observation
        self.obs = self.new_state
        # Aditional info
        self.rewardPerStep+=reward
        print('-----------------------------------------------------------')
        print('Episode Step:      ' + str(self.episode_step)+ '\n' +
              'Reward:            '+ str(reward) + '\n' +
              'Acomulated reward: ' + str(self.rewardPerStep)+'\n' +
              'Episode:           ' + str(self.epidose))
        print('-----------------------------------------------------------')
        #print(self.rewardPerStep)

        return np.array(self.obs, dtype=np.float32) , reward, endEpisode, info
    
    # ---------------------------------------------------------
    # ----------------------- Methods -----------------------
    # ---------------------------------------------------------
    def set_random_objetive(self):
        #Setting random objetive
        new_position_objetiveX = np.random.randint(-24,25)
        new_position_objetivey = np.random.randint(13,24)
        positionForObjetive = [new_position_objetiveX/10, new_position_objetivey/10,0]
        res= sim.simxSetObjectPosition(self.client, self.Sink, -1,positionForObjetive, sim.simx_opmode_oneshot)
        #Setting random position for quad 
        new_position_objetiveX = np.random.randint(-24,25)
        new_position_objetivey = np.random.randint(-22,-18.5)
        positionForQuad = [new_position_objetiveX/10, new_position_objetivey/10,0.51]
        res= sim.simxSetObjectPosition(self.client, self.base, -1,positionForQuad, sim.simx_opmode_oneshot)
        res= sim.simxSetObjectPosition(self.client, self.target_hand, -1,positionForQuad, sim.simx_opmode_oneshot)

    #  ----------------------- ACTION  ------------------------
    def movement_action(self, action, distance=0.03):
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
        while ((abs(prevX-self.pos[0]) < 0.02) and
                (abs(prevY-self.pos[1]) < 0.02) and ((time.time() - start_moving) < 0.1)):

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
        Izq,Frent,Dere,atras = self.get_close_to_obs()
        self.obs[2] = Izq
        self.obs[3] = Frent
        self.obs[4] = Dere
        self.obs[5] = atras
        contador_estrellos,a,b,c = self.if_get_coll_to_obs()
        self.obs[6] =contador_estrellos
        self.obs[7] = self.objetive[0]
        self.obs[8] = self.objetive[1]
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
        # directory = r'C:\Users\kscer\Documents\Universidad\Semestre 8\Tesis\Scripts'
        # os.chdir(directory)
        # cv2.imwrite('Prueba5.png', img)

        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray_image
    
    def hallarPuntoMedio(self,lado):
        imagen = self.get_image()
        puntoXsup= -1
        puntoXInf= -1
        puntoYsup= -1
        puntoYinf= -1
        puntoX = -1
        puntoY = -1
        if lado == 'I':
            for i in range(len(imagen)):
                for j in range(len(imagen[0])):
                    if 0<=j<=105:
                        if i+20<128 and j+20<256:
                            if (imagen[i][j]!=255 and imagen[i][j]>0) and (imagen[i][j+20]!=255 and imagen[i][j+20]>0) and (imagen[i+20][j+20]!=255 and imagen[i+20][j+20]>0) and (imagen[i+20][j]!=255 and imagen[i+20][j]>0):                 
                                
                                puntoXsup = j+20
                                puntoYsup = i
                                puntoYinf=  i+20
                                puntoXInf = j
                            
            puntoX = puntoXsup -10
            puntoY = puntoYsup- 10
        if lado == 'D':
            for i in range(len(imagen)):
                for j in range(len(imagen[0])):
                    if 105<j<=256:
                        if i+20<128 and j+20<256:
                            if (imagen[i][j]!=255 and imagen[i][j]>0) and (imagen[i][j+20]!=255 and imagen[i][j+20]>0) and (imagen[i+20][j+20]!=255 and imagen[i+20][j+20]>0) and (imagen[i+20][j]!=255 and imagen[i+20][j]>0):                 
                                
                                puntoXsup = j+20
                                puntoYsup = i
                                puntoYinf=  i+20
                                puntoXInf = j
                                
            puntoX = puntoXsup -10
            puntoY = puntoYsup- 10
        else:
            for i in range(len(imagen)):
                for j in range(len(imagen[0])):
                    if 100<j<=156:
                        if i+20<128 and j+20<256:
                            if (imagen[i][j]!=255 and imagen[i][j]>0) and (imagen[i][j+20]!=255 and imagen[i][j+20]>0) and (imagen[i+20][j+20]!=255 and imagen[i+20][j+20]>0) and (imagen[i+20][j]!=255 and imagen[i+20][j]>0):                 
                                
                                puntoXsup = j+20
                                puntoYsup = i
                                puntoYinf=  i+20
                                puntoXInf = j
                                
            puntoX = puntoXsup -10
            puntoY = puntoYsup- 10
            
            
        #print(imagen[puntoY][puntoX])
        return (puntoY,puntoX)# YA VIENE EN FORMATO PARA IMAGEN
    
    def get_close_to_obs(self):
        ancho = 256
        
        #alto = 128
       
        gray_image = self.get_image()
        contador = 0
        umbral = 800

        centinela1 = False
        centinela2 = False
        centinela3 = False
        
        disFrente = -1
        disIzquierda = -1
        disDerecha = -1
        disAtras = -1
        
        #y = 2x-16 Recta que describe la relacion entre la distancia y la intensidad de los pixeles
        #Examinar parte izquierda
        contador_izquierda = 0
        for fila in range(len(gray_image)):
            for columna in range(len(gray_image[0])):
                if 0<=columna<ancho-156:
                    if gray_image[fila][columna]<=self.tolUmbral5 and gray_image[fila][columna]>0:
                        contador_izquierda+=1
                            
    
        #Examinar parte centro
        contador_centro=0
        for fila in range(len(gray_image)):
            for columna in range(len(gray_image[0])):
                if (ancho-156)<=columna<(ancho-100):
                    if gray_image[fila][columna]<=self.tolUmbral5 and gray_image[fila][columna]>0:
                        contador_centro+=1
        
        #Examinar parte derecha
        contador_derecha=0
        for fila in range(len(gray_image)):
            for columna in range(len(gray_image[0])):
                if (ancho-100)<=columna<ancho:
                   if gray_image[fila][columna]<=self.tolUmbral5 and gray_image[fila][columna]>0:
                        contador_derecha+=1
        UmbralFrenteMax = 1000
        UmbralFrenteMin = 4523     
        umbral_centro = 800         
        if contador_izquierda>=UmbralFrenteMax:
            centinela1= True
        if contador_centro >umbral_centro:
            i,j= self.hallarPuntoMedio('')
            disAtras = gray_image[i][j]
            centinela2= True
        if contador_derecha>=UmbralFrenteMax:
            centinela3 = True
        
        #print(centinela1,centinela2, centinela3, contador_izquierda,contador_centro,contador_derecha)
        
        if centinela1 and centinela3 and centinela2== False:
                if 2822<=contador_derecha<=6700 and 2822<=contador_izquierda<=6700:
                    iIz,jIz = self.hallarPuntoMedio('I')
                    iDe,jDe = self.hallarPuntoMedio('D')
                    disDerecha = gray_image[iDe][jDe]
                    disIzquierda = gray_image[iIz][jIz]
                    
                else:
                    disFrente= gray_image[60][5]
                if contador_derecha> 4168 and contador_izquierda<4168:
                    disIzquierda = -1
                    disDerecha = gray_image[60][200]
                elif contador_izquierda>4168 and contador_derecha<4523:
                    disDerecha = -1
                    disIzquierda= gray_image[60][40]
                elif contador_izquierda>4168 and contador_derecha>4168:
                    disIzquierda= gray_image[60][40]
                    disDerecha = gray_image[60][200]
        
        else:
            if centinela1:
                lado='I'
                i,j = self.hallarPuntoMedio(lado)
                disIzquierda = gray_image[i][j]
            elif centinela3:
                lado='D'
                i,j = self.hallarPuntoMedio(lado)
                disDerecha = gray_image[i][j]
    
        if disFrente != -1 and disFrente<=28:    
            disFrente = (disFrente*2-16)/100
        if disIzquierda != -1 and disIzquierda<=28:  
            disIzquierda = (disIzquierda*2-16)/100
        if disDerecha != -1 and disDerecha<=28:  
            disDerecha = (disDerecha*2-16)/100
        if disAtras != -1 and disAtras<=28:  
            disAtras = (disAtras*2-16)/100
            
        return disIzquierda,disFrente,disDerecha,disAtras

    #  ------------------------ REWARD  ------------------------
    
    def if_get_coll_to_obs(self):

       
        ancho = 256
        
        #alto= 128

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
        info  = []
        goalReached = False
        outOfBounds = False
        lostForTooLong = False
        collision = False
        maxItReached = False

        # Reward for getting closer to an obstacule
        if 0<newState[2]<=0.3:
            reward-= (1/newState[2])*10

        if 0<newState[3]<=0.3:
            reward-= (1/newState[3])*10
            
        if 0<newState[4]<=0.3:
            reward-= (1/newState[4])*10
        if 0<newState[5]<=0.3:
            reward-= (1/newState[5])*10
        print('Recompensa: ', reward)
        print(newState[2],newState[3],newState[4],newState[5])
        #reward for getting close to the objetive
        a= np.array((newState[0],newState[1]))
        b= np.array((newState[7],newState[8]))
        self.prevEuclidianPos =  self.nextEuclidianPos
        self.nextEuclidianPos = np.linalg.norm(a-b)
        #print('Distancia previa: ', self.prevEuclidianPos, 'distancia actual:', self.nextEuclidianPos)
        if self.nextEuclidianPos<self.prevEuclidianPos:
            #print('Entro a 50')
            reward += 25
        elif self.nextEuclidianPos>self.prevEuclidianPos:
            #print('Entro a 25')
            reward += -75
        # Update position
        
        # Verify if robot reach goalReached()
        
        if  ((newState[7]-0.15)<=newState[0]<=newState[7]+0.15) and ((newState[8]-0.15)<=newState[1]<=newState[8]+0.15):
        
            goalReached = True
            print('EPISODE TERMINATED: Robot reached goal')
            info = ['1' ,'reached goal']
            reward += 1000 - step
        
        # Verify if robot out of bounds
        if (newState[0] > MAX_X-0.05) or (newState[1] > MAX_Y-0.05) or (newState[0] < MIN_X+0.05) or (newState[1] < MIN_Y+0.05):
            outOfBounds = True
            reward += -1000
            print('EPISODE TERMINATED: Robot out of bounds')
            info = ['2', 'out of bounds']
  
        # Verify if collision with obstacle
        obst_counter = newState[6]
        if obst_counter>0:
            collision = True
            reward += -1000
            print('EPISODE TERMINATED: Robot collide with an obstacle')
            info = ['3', 'collide with an obstacule']
            
        # Verify if max iterations reached
        if (step > self.max_it):
            maxItReached = True
            print('EPISODE TERMINATED: Max iterations reached')
            info = ['4', 'max iterations']

        info = {}
        # End episode condition
        endEpisode = maxItReached or outOfBounds or goalReached or lostForTooLong or collision
        return (reward/100), endEpisode,info
        
        
    
def main():
    

    
    path_tensorBoard = 'C:/Users/kscer/Documents/Universidad/Semestre 8/Tesis/results/logs/' 
    #tensorboard --logdir ./ --host=127.0.0.1
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
    cnn_model = DQN(MlpPolicy, env, tensorboard_log= path_tensorBoard, learning_rate=0.0001, buffer_size=1000000,
                    learning_starts=50000, batch_size=32, optimize_memory_usage=False, target_update_interval=5000,
                    exploration_fraction=0.1, exploration_initial_eps=1.0, exploration_final_eps=0.025, verbose=1)
    
    
    
    # Training

    cnn_model.learn(total_timesteps=700000, callback=callback, tb_log_name= datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    cnn_model.save('C:/Users/kscer/Documents/Universidad/Semestre 8/Tesis/results/logs/EntrenamientoCompletado/'+ datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))






if __name__ == "__main__":
    # Main function
    #from stable_baselines3.common.env_checker import check_env
    #env = quadEnviSim()
    #check_env(env, warn=True)
    main()    
        
        
        
        
        
        
        
        
    
