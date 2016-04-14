import tensorflow as tf
import numpy as np
import time
import os
import logger
import matplotlib.pyplot as plt
import cv2 as cv
import cPickle as pickle

DQN_TRAINING = True
TEACHING = False

class DQN():

	def __init__(self,
	             training_start_frame = 10000,
	             trianing_end_frame   = 2000000,
	             random_pick_p_start = 1.0,
	             random_pick_p_end = 0.05,
	             random_pick_peiriod = 80000,
	             batch_size = 4,
	             image_size = 84,
	             input_channels = 4,
	             minibatch_size=32,
	             discount_rate=0.9,
	             first_conv_stride = 4,
	             first_conv_patch_size = 8,
	             first_conv_out_depth = 16,
	             second_conv_stride = 2,
	             second_conv_patch_size = 4,
	             second_conv_out_depth = 32,
	             first_fully_connect_layer_size = 256,
	             second_fully_connect_layer_size = 256,
	             single_state_size = 1,
	             RMSProp_learing_rate = 0.0000001,
	             output = 5,  # up down left right fire
	             checkpoint_save_frequnce = 5000,
					target_network_update_rate=0.01
	             ):

		self.graph = tf.get_default_graph()
		#self.sess = tf.Session(graph=self.graph)
		self.sess = tf.InteractiveSession()
		#self.optimizer = tf.train.RMSPropOptimizer(learning_rate= 0.001, decay=0.9)
		self.training_start_frame = training_start_frame
		self.trianing_end_frame = trianing_end_frame
		self.batch_size = batch_size
		self.image_size = image_size
		self.in_c = input_channels
		self.minibatch_size = minibatch_size
		self.discount_rate = discount_rate
		self.q_real = np.array(np.zeros((self.minibatch_size)),dtype=float)
		self.conv1_stride = first_conv_stride
		self.conv1_p_size = first_conv_patch_size
		self.conv1_depth = first_conv_out_depth
		self.conv2_stride = second_conv_stride
		self.conv2_p_size = second_conv_patch_size
		self.conv2_depth = second_conv_out_depth
		self.fc1_size = first_fully_connect_layer_size
		self.second_fully_connect_layer_size = second_fully_connect_layer_size
		self.output_num = output
		self.single_state_size = single_state_size
		self.random_pick_p_start = random_pick_p_start
		self.random_pick_p_end = random_pick_p_end
		self.random_action_porb = random_pick_p_start
		self.random_pick_peiriod = random_pick_peiriod
		self.RMSProp_learing_rate = RMSProp_learing_rate
		self.create_model()
		self.loaded = True
		self.error, self.optimizer_op, self.y = self.training_computation()
		self.error_list = []
		self.checkpoint_save_frequnce = checkpoint_save_frequnce
		'''
		self.saver = tf.train.Saver(var_list={'l1_w': self.layer1_weights,
		                            'l1_b': self.layer1_biases,
		                            'l2_w': self.layer2_weights,
		                            'l2_b': self.layer2_biases,
		                            'l3_w': self.layer3_weights,
		                            'l3_b': self.layer3_biases,
		                            'l4_w': self.layer4_weights,
		                            'l4_b': self.layer4_biases},
		                            max_to_keep=2)
		'''

		self.saver = tf.train.Saver()


		self.sess.run(tf.initialize_all_variables())


	def create_model(self):

		#input data
		self.actions = tf.placeholder(tf.int8, shape=(self.minibatch_size),name='actions')
		self.rewards = tf.placeholder(tf.float32, shape=(self.minibatch_size),name='rewards')
		self.terminal = tf.placeholder(tf.float32, shape=(self.minibatch_size),name='terminal')
		self.y = tf.placeholder(tf.float32, shape=(self.minibatch_size),name='y')

		'''
		self.s = tf.placeholder(tf.float32, shape=(
			self.minibatch_size, self.image_size, self.image_size, self.input_channels),name='s0')
		'''
		self.s = tf.placeholder(tf.float32, shape=(
			None, self.image_size, self.image_size, self.in_c), name='s0')

		self.current_screen = tf.placeholder(tf.float32, shape=(
			self.single_state_size, self.image_size, self.image_size, self.in_c), name='current_screen')

		self.discount_q_feed = tf.placeholder(tf.float32, shape=(self.minibatch_size),name='discount_q_feed')

		self.q_s0 = tf.placeholder(tf.float32, shape=(self.minibatch_size),name='y')

		'''
		Variables:
		'''

		self.l1_w = tf.Variable(tf.truncated_normal(
          [self.conv1_p_size, self.conv1_p_size, self.in_c, self.conv1_depth],
			stddev=0.1), name='l1_w')

		self.l1_b = tf.Variable(tf.zeros([self.conv1_depth]), name='l1_b')

		self.l2_w = tf.Variable(tf.truncated_normal(
          [self.conv2_p_size, self.conv2_p_size, self.conv1_depth, self.conv2_depth],
			stddev=0.1), name='l2_w')
		self.l2_b = tf.Variable(tf.constant(1.0, shape=[self.conv2_depth]), name='l2_b')

		l3_in_size = 9 * 9 * self.conv2_depth
		self.fc1_w = tf.Variable(tf.truncated_normal(
          [l3_in_size, self.fc1_size], stddev=0.1), name='l3_w')
		self.fc1_b = tf.Variable(tf.constant(1.0, shape=[self.fc1_size]), name='l3_b')

		self.fc2_w = tf.Variable(tf.truncated_normal(
          [self.fc1_size, self.output_num], stddev=0.1), name='l4_w')

		self.fc2_b = tf.Variable(tf.constant(1.0, shape=[self.output_num]), name='l4_b')

		'''
		Deep network model
		'''
		#norm_s = tf.nn.l2_normalize(self.s, 1, epsilon=1e-12, name=None)

		conv = tf.nn.conv2d(self.s, self.l1_w,
		                    [1, self.conv1_stride, self.conv1_stride, 1],
		                    padding='VALID')
		hidden = tf.nn.relu(conv + self.l1_b)

		conv = tf.nn.conv2d(hidden, self.l2_w,
		                    [1, self.conv2_stride, self.conv2_stride, 1],
		                    padding='VALID')
		hidden = tf.nn.relu(conv + self.l2_b)
		shape = hidden.get_shape().as_list()

		reshape = tf.reshape(hidden, [-1, 9*9*32])
		#reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])

		hidden = tf.nn.relu(tf.matmul(reshape, self.fc1_w) + self.fc1_b)

		self.output = tf.matmul(hidden, self.fc2_w) + self.fc2_b


	def choose_action(self, current_screen, total_frame):

		start_time = time.clock()

		if total_frame > self.training_start_frame:
		#p_initial - (n * (p_initial - p_final)) / (total)
			if self.random_action_porb > self.random_pick_p_end:
				self.random_action_porb -= \
					(self.random_pick_p_start - self.random_pick_p_end) / self.random_pick_peiriod
		else:
			self.random_action_porb = self.random_pick_p_start


		logger.log_write_info('random_action_porb = ' + str(self.random_action_porb))

		if np.random.rand() < self.random_action_porb:
			nextaction = np.random.randint(0, 5)
			logger.log_write_debug(' dqn, choose action rondomly, need time '
			                       + str(time.clock() - start_time))
			logger.log_write_info('random action ' + str(nextaction))
			return [nextaction]
		else:
			q = self.output.eval(feed_dict={self.s : [current_screen]})
			logger.log_write_info(q)
			nextaction = np.argmax(q, 1)
			logger.log_write_info('dqn select action ' + str(nextaction))
			logger.log_write_debug(' dqn, choose action by DQN, need time '
			                       + str(time.clock() - start_time))
			return nextaction

	def training_computation(self):


		y = self.rewards + tf.mul(self.terminal, tf.reduce_max(self.output, reduction_indices = 1))

		error = tf.reduce_mean(tf.square(tf.sub(y, self.q_s0)))

		optimizer_op = tf.train.RMSPropOptimizer(learning_rate=self.RMSProp_learing_rate, decay=0.9).minimize(error)
		#optimizer_op = tf.train.AdamOptimizer(1e-6).minimize(error)

		return error, optimizer_op , y


	def dqn_training(self, db_manager, frame_num):

		if os.listdir('./ckp') != []:
			if self.loaded == False:
				self.load_checkpoint()
			self.load_memory = False

		s0, actions, rewards, s1, terminal, frames = db_manager.get_minibach_smaple()
		'''
		y = []

		for i in range(0, len(self.minibatch_size)):
			if terminal[i]:
				y.append(rewards[i])
			else:
				y.append(rewards[i] + self.discount_rate * np.max(self.model(s1[i])))
		'''

		q = self.output.eval(feed_dict={self.s : s0})

		for i in range(self.minibatch_size):
			l = q[i]
			a = actions[i]
			if a == -1 or a == None:
				a = 4
			self.q_real[i] = l[a]

		logger.log_write_info('q real = ' + str(self.q_real))

		op, y, error, l1_w = self.sess.run([self.optimizer_op, self.y, self.error, self.l1_w],
		                                   {
					self.s : s1,
					self.terminal : terminal,
					self.q_s0 : self.q_real,
					self.rewards : rewards
				})

		self.error_list.append(error)
		logger.log_write_info('error ' + str(error))
		logger.log_write_info('terminal ' + str(terminal))

		logger.log_write_info(' y = ' + str(y))

		if frame_num > 3000 and frame_num % self.checkpoint_save_frequnce == 0:
			print 'check point saved '
			logger.log_write_info('check point saved ')
			self.saver.save(sess=self.sess,
			                save_path='./ckp/dqn_' + str(frame_num))
			self.draw_learning_curve(frame_num, l1_w)

		#logger.log_write_info('q_result %f' + str(self.q_real))
		#logger.log_write_info('q_max %f' +str(q_max))
		#logger.log_write_info('y %f' +str(y))
		#logger.log_write_info('frames %f' +str(frames))
		#logger.log_write_info('actions %f' +str(actions))
		#logger.log_write_info('training error  = ' + str(error))

		#cv.imwrite('./samples/img_' + str(frame_num) + '.png', dst)

		#print 'layer1_weights ', layer1_weights
		#print 'layer2_weights ', layer2_weights
		#print 'layer3_weights ', layer3_weights
		#print 'layer4_weights ', layer4_weights

	def load_checkpoint(self):
		self.saver.restore(sess=self.sess, save_path='./ckp/dqn')
		print('***************************check point loaded*******************************************')
		logger.log_write_info('***************************check point loaded*******************************************')
		self.loaded = True

	def draw_learning_curve(self, frame, l1_w):
		for i in range(l1_w.shape[3]):
			for j in range(l1_w.shape[2]):
				cv.imwrite('./vis/filter' + str(i) + '_' + str(j) + '.png', 255 * l1_w[:,:,j,i])

		plt.figure(0)
		xaxis = np.arange(0, len(self.error_list))
		plt.plot(xaxis, self.error_list, label='cross set predictions')
		plt.savefig('myfig' + str(frame) +'.png')
