import cv2 as cv
import os
import numpy as np
import time
import logger
import random
from collections import deque
import cPickle as pickle
import copy

class Data_manager():

	def __init__(self):
		self.temp_state = None
		#self.current_observation = []
		#self.current_observation = np.array(np.zeros((84,84,4)))
		#self.following_observation = np.array(np.zeros((84,84,4)))
		self.state_length = 4
		self.current_action = -1
		self.current_reward = 0

		self.minibatch_length = 32
		self.current_init = 0
		self.memory_size = 10000
		self.memory_zero = deque()
		self.memory_nonzero = deque()
		img_init_array = np.array(np.zeros((84,84)),dtype=float)
		self.current_observation = np.stack((img_init_array, img_init_array, img_init_array, img_init_array), axis=2)
		self.following_observation = np.stack((img_init_array, img_init_array, img_init_array, img_init_array), axis=2)

		self.start_frame = True

		self.load_memory = False
		self.pickle_name = './memory.pickle'
		self.pickle_saved = False

	def save(self,frame_num, img, action, reward, terminal):

		if self.load_memory == True:
			with open(self.pickle_name, 'rb') as f:
				m = pickle.load(f)
				self.memory_nonzero = m['nz']
				self.memory_zero = m['zz']
			self.load_memory = False

		if terminal == 1:
			if frame_num % 8 != 0:
				return

		'''

		for p in range(img.shape[0]):
			print '****************************'
			print p[:20,:20, r]
			print p[21:41,21:41, r]
			print p[42:62,42:62, r]
			print p[63:84,42:84, r]
			print '****************************'
		'''

		resized_img = cv.resize(img, dsize=(84,84))

		resized_img = np.reshape(resized_img, (84,84,1))

		'''
		if frame_num % 10000 == 0:
			for i in range(4):
				cv.imwrite('./samples/c_' + str(frame_num) + '_' + str(i) + '_c.png', self.current_observation[:,:,i ])
		'''
		self.following_observation = np.append(resized_img, self.current_observation[:, :, :3], axis=2)

		'''
		if frame_num % 10000 == 0:
			for i in range(4):
				cv.imwrite('./samples/f_' + str(frame_num) + '_' + str(i) + '.png', self.following_observation[:,:,i ])
		'''

		if terminal == 1:
			self.memory_zero.append((self.current_observation, action, reward, self.following_observation, terminal, frame_num))
		else:
			logger.log_write_info(' get a non zero state ' + str(terminal))
			self.memory_nonzero.append((self.current_observation, action, reward, self.following_observation, terminal, frame_num))

		self.current_observation = self.following_observation

		if len(self.memory_nonzero) + len(self.memory_zero) == self.memory_size:
			if random.random < 0.05:
				self.memory_nonzero.popleft()
			else:
				self.memory_zero.popleft()
			if self.pickle_saved == False:
				with open(self.pickle_name, 'wb') as f:
					pickle.dump({'nz':self.memory_nonzero, 'z':self.memory_zero}, f, protocol=2)
				print 'pickle saved'
				self.pickle_saved = True

		if terminal == 0:
			img_init_array = np.array(np.zeros((84,84)),dtype=float)
			self.current_observation = np.stack((img_init_array, img_init_array, img_init_array, img_init_array), axis=2)

	def get_minibach_smaple(self):

		non_zero = random.randint(0, min(self.minibatch_length, len(self.memory_nonzero)))
		minibatch = random.sample(self.memory_nonzero, non_zero) + random.sample(self.memory_zero, self.minibatch_length - non_zero)

		minibatch_s0 = [d[0] for d in minibatch]
		minibatch_actions = [d[1] for d in minibatch]
		minibatch_rewards = [d[2] for d in minibatch]
		minibatch_s1 = [d[3] for d in minibatch]
		minibatch_term = [d[4] for d in minibatch]
		minnibatch_frames = [d[5] for d in minibatch]

		'''
		i = 0
		for p in minibatch_s1:
			cv.imwrite('./samples/' + str(i) + '_0_f.png', p[:,:,0 ])
			cv.imwrite('./samples/' + str(i) + '_1_f.png', p[:,:,1 ])
			cv.imwrite('./samples/' + str(i) + '_2_f.png', p[:,:,2 ])
			cv.imwrite('./samples/' + str(i) + '_3_f.png', p[:,:,3 ])
			i += 1
		'''


		return minibatch_s0, \
		       minibatch_actions, \
		       minibatch_rewards, \
		       minibatch_s1, \
		       minibatch_term, \
		       minnibatch_frames

	def get_current_observation(self):
		return self.following_observation