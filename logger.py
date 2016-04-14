import logging
import os
import time

file_name = ''

def init_logger():
	t = time.localtime()
	filename = './Log/'+ str(t[0])+ '_' + str(t[1])+ '_' +str(t[2]) \
	           + '_' + str(t[3]) + '_' + str(t[4]) + '.log'
	logging.basicConfig(filename=filename,level=logging.DEBUG)

def log_write_info(s):
	logging.info(s)

def log_write_debug(s):
	logging.debug(s)

'''
logging.basicConfig(filename='example.log',level=logging.DEBUG)
logging.debug('This message should go to the log file')
logging.info('So should this')
logging.warning('And this, too')
'''