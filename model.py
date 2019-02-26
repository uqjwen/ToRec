import sys
import os
import numpy as np 
import tensorflow as tf 
import keras
from keras import backend as K
from keras.regularizers import l1, l2
from keras.models import Sequential, Model
from keras.models import load_model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, merge, Reshape, Merge, Flatten, Dropout
from keras.layers import Lambda
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
import pickle
from time import time
# from metrics import precision_k_curve,recall_k_curve,ndcg_k_curve
from utils import precision_k_curve, recall_k_curve, ndcg_k_curve,hr_k_curve, cos_sim, map_k_curve
from tensorflow.contrib import rnn
from tensorflow.python.ops import control_flow_ops  
from tensorflow.python.training import moving_averages  
from tensorflow.python.training.moving_averages import assign_moving_average



class Model():

	def __init__(self,batch_size,layers,
				num_user,num_item,hidden_size, f_matrix, n_neighbor_1, n_neighbor_2):
		print("Building model...")
		self.num_user = num_user
		self.num_item = num_item

		self.user_input = tf.placeholder(tf.int32,shape=[None])#shape=None
		self.item_input = tf.placeholder(tf.int32, shape=[None])

		self.prediction = tf.placeholder(tf.float32, shape=[None,1])




		self.user_embedding = tf.Variable(
			tf.random_uniform([num_user, hidden_size],-1.0,1.0))


		self.item_embedding = tf.Variable(
			tf.random_uniform([num_item, hidden_size],-1.0,1.0))

		# self.mlp_user_embedding = tf.Variable(
		# 	tf.random_uniform([num_user, hidden_size], -1.0,1.0))
		# self.mlp_item_embedding = tf.Variable(
		# 	tf.random_uniform([num_item, hidden_size],-1.0,1.0))

		
		
		user_latent = tf.nn.embedding_lookup(self.user_embedding, self.user_input)
		item_latent = tf.nn.embedding_lookup(self.item_embedding, self.item_input)

		# mlp_user_latent = tf.nn.embedding_lookup(self.mlp_user_embedding, self.user_input)
		# mlp_item_latent = tf.nn.embedding_lookup(self.mlp_item_embedding, self.item_input)


		sim1 = tf.reduce_sum(tf.multiply(user_latent, item_latent), axis=1, keep_dims = True)
		vector1 = tf.concat([user_latent, item_latent,sim1, tf.multiply(user_latent, item_latent) ], axis=1)
		vector1 = tf.layers.batch_normalization(vector1)

		for i in range(len(layers)):
			hidden = Dense(layers[i], activation='relu',kernel_initializer = 'lecun_uniform',name='v1_ui_hidden_' + str(i))
			vector1 = hidden(vector1)
		self.logits = Dense(1, kernel_initializer='lecun_uniform', name = 'prediction')(vector1)
		# self.logits = Dense(1, name = 'prediction')(vector1)

		# mf_vector = tf.multiply(user_latent, item_latent)

		# sim = tf.reduce_sum(tf.multiply(mlp_user_latent, mlp_item_latent),axis=1, keep_dims=True)
		# mlp_vector = tf.concat([mlp_user_latent, mlp_item_latent,sim],axis=1)
		# # mlp_vector = tf.layers.batch_normalization(mlp_vector)
		# for i in range(len(layers)):
		# 	hidden = Dense(layers[i], kernel_initializer = 'lecun_uniform',activation='relu', name="layer%d"%i)
		# 	mlp_vector = hidden(mlp_vector)

		# # self.logits = self.logits_1
		# # self.logits = tf.cond(self.texting, lambda:temp_logits+self.logits_2, lambda:temp_logits)
		# predict_vector = tf.concat([mf_vector, mlp_vector], axis=1)
		# self.logits = Dense(1,kernel_initializer='lecun_uniform', name='prediction')(predict_vector)
		att1 = self.attentive_social(self.user_input, item_latent, self.user_embedding, n_neighbor_1, 'att1')

		# att2 = self.attentive_social(self.user_input, item_latent, self.user_embedding, n_neighbor_2, 'att2')
		att2 = self.attentive_social(self.item_input, user_latent, self.item_embedding, n_neighbor_2, 'att2')




		self.pred = tf.nn.sigmoid(self.logits)





		# self.loss = tf.reduce_mean(tf.sigmoid_cross_entropy_with_logits(logits,prediction))	
		self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.logits, labels = self.prediction))



		reg_error = tf.nn.l2_loss(self.user_embedding)+\
					tf.nn.l2_loss(self.item_embedding)
					# tf.nn.l2_loss(self.mlp_item_embedding)+\
					# tf.nn.l2_loss(self.mlp_user_embedding)
		self.cost = self.loss
		self.cost += 0.0001*reg_error
		self.cost += 0.001*self.regularization(f_matrix, self.user_input, self.user_embedding)
		# self.cost += 0.001*self.attentive_social(self.user_input, item_latent, self.user_embedding, n_neighbor)
		self.cost += float(sys.argv[2])*self.hybrid_attention(user_latent, att1, att2)

		self.train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.cost)
		# self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)
	def hybrid_attention(self, user_latent, att1, att2):
		att1 = tf.expand_dims(att1,1)
		att2 = tf.expand_dims(att2,1)
		hybrid = tf.concat([att1,att2], axis=1)

		alpha_layer = Dense(1,kernel_initializer = 'lecun_uniform', name = 'hybrid_att')
		logits = alpha_layer(hybrid)  ####batch 2 1 
		logits = tf.reshape(logits, [-1,2])

		self.alpha = tf.nn.softmax(10*logits)

		alpha = tf.expand_dims(self.alpha, -1)

		hybrid = tf.reduce_sum(alpha*hybrid, axis=1) ##### batch hidden_dima

		reg = tf.nn.l2_loss(user_latent - hybrid)

		return reg





	def attentive_social(self,user_input, item_latent, user_embedding, n_neighbor, name):
		embedding_size = item_latent.shape.as_list()[-1]
		neighbors = tf.nn.embedding_lookup(n_neighbor, user_input) ##batch * 20
		num_neighbors = n_neighbor.shape[-1]
		neighbor_embedding = tf.nn.embedding_lookup(user_embedding, neighbors) ## batch *20 * embed_size


		expand_item_latent = tf.expand_dims(item_latent, 1)
		expand_user_latent = tf.expand_dims(tf.nn.embedding_lookup(user_embedding, user_input),1)


		hybrid_item_neighbor = tf.multiply(neighbor_embedding, expand_item_latent)

		# u_layer = Dense(embedding_size,kernel_initializer = 'lecun_uniform', name = 'u_layer')
		# i_layer = Dense(embedding_size,kernel_initializer = 'lecun_uniform', name = 'i_layer')
		# ui_layer = Dense(embedding_size,kernel_initializer = 'lecun_uniform', name = 'ui_layer')
		# hybrid = tf.nn.tanh(u_layer(neighbor_embedding)+i_layer(expand_item_latent)+ui_layer(hybrid_item_neighbor))
		# vector = Dense(1,kernel_initializer = 'lecun_uniform', name='vector')
		# hybrid = vector(hybrid)
		# hybrid = tf.reshape(hybrid, [-1,num_neighbors])

		units = int(embedding_size/2)
		layer_item_1 = Dense(units, activation='relu',kernel_initializer = 'lecun_uniform',name=name+'layer_item_1')
		layer_item_2 = Dense(1, kernel_initializer = 'lecun_uniform', name=name+'layer_item_2')
		hybrid_item = layer_item_2(layer_item_1(hybrid_item_neighbor))
		hybrid = tf.reshape(hybrid_item, [-1, num_neighbors])

		# hybrid = tf.reduce_sum(hybrid_item_neighbor, axis=-1)		
		if name == 'att1':
			self.att1 = tf.nn.softmax(10*hybrid)      ## attentions for visualization 
			hybrid = tf.expand_dims(self.att1,-1) ###batch 20 1
		else:
			self.att2 = tf.nn.softmax(10*hybrid)
			hybrid = tf.expand_dims(self.att2, -1)

		att_neighbor = tf.reduce_sum(tf.multiply(neighbor_embedding, hybrid), axis = 1)

		return att_neighbor

		user_embedding = tf.nn.embedding_lookup(user_embedding, user_input)

		reg = tf.reduce_sum(tf.square(user_embedding - att_neighbor))

		return reg





	def regularization(self,f_matrix, user_input, user_embedding):

		batch_f_matrix = tf.nn.embedding_lookup(f_matrix, user_input)
		sum_bfm = tf.reduce_sum(batch_f_matrix, axis = 1, keep_dims = True)
		n_sum_bfm = batch_f_matrix/(sum_bfm+10e-10)
		ref_f = tf.matmul(n_sum_bfm, user_embedding)


		batch_embedding = tf.nn.embedding_lookup(user_embedding, user_input)
		reg = tf.reduce_sum(tf.square(batch_embedding - ref_f))
		return reg

		



class Data_Loader():
	def __init__(self, batch_size):
		print("data loading...")
		pickle_file = open("data.pkl",'rb')
		# pickle_file = open('data40.pkl','rb')



		self.data = pickle.load(pickle_file)
		self.R_m = self.data['ratings']
		self.num_user = self.data['num_user']
		self.num_item = self.data['num_item']
		self.batch_size = batch_size
		# self.f_matrix = self.data['f_matrix_2'].astype(np.float32)
		self.f_matrix = np.load('train_s_mat.npy').astype(np.float32)
		# self.n_neighbor = self.data['n_neighbor']
		self.n_neighbor_1 = np.load('train_s_neighbor.npy')
		# self.n_neighbor_s = np.load('train_s_neighbor_s.npy').astype(np.float32)
		self.n_neighbor_2 = np.load('item_s_neighbor.npy')


	def reset_data(self):

		print("resetting data...")
		u_input = self.data['train_user'][:]
		i_input = self.data['train_item'][:]
		item_num = self.data['num_item']
		# ui_label = self.data['train_label'][:]

		u_input = []
		i_input = []
		ui_label = []
		negative_samples_num = 5


		for u,i in zip(self.data['train_user'], self.data['train_item']):
			u_input.append(u)
			i_input.append(i)
			ui_label.append(1)
			all_item = range(item_num)
			# missing_values = list(set(all_item)-set(self.R_m[u]))

			u_input.extend([u]*negative_samples_num)
			# negative_samples = np.random.choice(missing_values,negative_samples_num, replace=False)
			negative_samples = np.random.randint(0,item_num, negative_samples_num)
			i_input.extend(list(negative_samples))
			ui_label.extend([0]*negative_samples_num)


		p_index = np.random.permutation(range(len(u_input)))
		self.u_input = np.array(u_input)[p_index]
		self.i_input = np.array(i_input)[p_index]
		self.ui_label = np.array(ui_label)[p_index]
		self.train_size = len(u_input)



	def reset_pointer(self):
		self.pointer = 0

	def next_batch(self):
		start = self.pointer*self.batch_size
		end = (self.pointer+1)*self.batch_size

		self.pointer+=1
		# return self.u_input[start:end], self.i_input[start:end], self.ui_label[start:end]
		item_index = self.i_input[start:end]
		user_index = self.u_input[start:end]



		return self.u_input[start:end],\
		self.i_input[start:end],\
		self.ui_label[start:end]








def get_data(user,item,data_loader):
	data = data_loader.data 
	# c_item = range(data['num_item'])
	num_item = data['num_item']
	negative_items = 100
	u = [user]*negative_items
	i = [item]+np.random.randint(0,num_item,(negative_items-1)).tolist()
	ui_label = [1]+[0]*(negative_items-1)
	pmtt = np.random.permutation(negative_items)
	return np.array(u),\
			np.array(i)[pmtt],\
			np.array(ui_label)[pmtt]


def test(data_loader, model):
	with tf.Session() as sess:


		load_all_variable(sess)		
		user_embedding, item_embedding = sess.run([model.user_embedding, model.item_embedding])
		np.save(checkpoint_dir+'user_embedding', user_embedding)
		np.save(checkpoint_dir+'item_embedding', item_embedding)
		# user_embedding, group_embedding = sess.run([model.user_embedding, model.group_embedding])
		# np.savetxt(checkpoint_dir+'user_embedding', user_embedding)
		# np.savetxt(checkpoint_dir+'group_embedding', group_embedding)


		res_matrix = [[],[]]
		max_k=10
		metrics_num = 2
		f = [hr_k_curve,ndcg_k_curve]
		count = 0
		fr = open(checkpoint_dir+ 'attentions','w')

		for user, item in zip(data_loader.data['test_user'], data_loader.data['test_item']):
			# u,i,u_text,i_text, item_adj,y_true = get_data(u,data_loader)
			u,i, ui_label = get_data(user,item,data_loader)

			y_pred,att = sess.run([model.pred, model.att1], feed_dict = {model.user_input:u,
														model.item_input:i,
														model.prediction:ui_label.reshape((-1,1))})


			att = np.around(att,decimals = 4)[0]
			# for item in att:
			fr.write(str(u[0])+':\t'+'\t'.join(map(str,att))+'\n')

			for i in range(metrics_num):
				res = f[i](ui_label.flatten(),y_pred.flatten(),max_k)
				res_matrix[i].append(res[:])


			count+=1
			if (count)%3000==0:
				print (np.mean(np.array(res_matrix),axis=1))
			sys.stdout.write("\ruser: "+str(count))
			sys.stdout.flush()
		print (np.mean(np.array(res_matrix),axis=1))
		fr.close()
		
		res = np.mean(np.array(res_matrix), axis=1).T
		np.savetxt(checkpoint_dir+"res.dat", res, fmt = "%.5f", delimiter = '\t')
		# f = open("res_tf_social.pkl",'wb')
		# pickle.dump(res_matrix,f)
		# f.close()
def sample(u,i):
	user = []
	item = []
	for usr,itm in zip(u,i):
		rand = np.random.random()
		if rand<0.8:
			user.append(usr)
			item.append(itm)
	return user,item

def val(data_loader,sess, model, tv_user, tv_item):
	res_matrix = [[],[]]
	max_k=10
	metrics_num = 2
	f = [hr_k_curve,ndcg_k_curve]
	for user, item in zip(tv_user,tv_item):
		# u,i,u_text,i_text, item_adj,y_true = get_data(u,data_loader)
		u,i, ui_label = get_data(user,item,data_loader)
		y_pred = sess.run([model.pred], feed_dict = {model.user_input:u,
													model.item_input:i,
													model.prediction:ui_label.reshape((-1,1))})
		for i in range(metrics_num):
			res = f[i](ui_label.flatten(),y_pred[0].flatten(),max_k)
			res_matrix[i].append(res[:])

	res = np.mean(np.array(res_matrix), axis=1).T
	return res[-1,0]

def load_all_variable(sess):
		saver = tf.train.Saver(tf.global_variables())

		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
			print (' [*] Loaded all parameters success!!!')
		else:
			print (' [!] Loaded all parameters failed...')


def load_part_variables(directory, sess):
	filename = directory+'v_name'
	fr = open(filename,'rb')
	v_name = pickle.load(fr)
	variables = [v for v in tf.global_variables() if v.name in v_name]
	saver = tf.train.Saver(variables)
	ckpt = tf.train.get_checkpoint_state(directory)
	if ckpt and ckpt.model_checkpoint_path:
		saver.restore(sess, ckpt.model_checkpoint_path)
		print(' [*] loading parital parameter success!!!')
	else:
		print(' [!] loading partial parameter failed...')

		load_all_variable(sess)


def train(batch_size,data_loader, model):
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		# load_part_variables('./model_so_reg/', sess)
		load_all_variable(sess)
		saver = tf.train.Saver(tf.global_variables())
		# save_dir = './'+sys.argv[0].split('.')[0][:-2]+'_'+sys.argv[2]+'/'
		# user_embedding = np.genfromtxt(save_dir+'user_embedding')
		# group_embedding = np.genfromtxt(save_dir+'group_embedding')
		# sess.run(tf.assign(model.user_embedding, user_embedding))
		# sess.run(tf.assign(model.group_embedding, group_embedding))
		# user_embedding = np.load('./neumf/user_embedding.npy')
		# item_embedding = np.load('./neumf/item_embedding.npy')
		# sess.run(tf.assign(model.user_embedding))
		# tv_user,tv_item = sample(data_loader.data['val_user'], data_loader.data['val_item'])
		tv_user,tv_item = sample(data_loader.data['test_user'], data_loader.data['test_item'])
		best_hr_10 = 0
		epochs_1 = 10
		epochs_2 = 50
		hr = []
		for i in range(epochs_1):
			data_loader.reset_data()
			total_batch = int(data_loader.train_size/batch_size)
			for e in range(epochs_2):
				data_loader.reset_pointer()
				for b in range(total_batch):
					iterations = i*epochs_2*total_batch+e*total_batch+b
					u_input, i_input, ui_label = data_loader.next_batch()
					train_loss, _ = sess.run([model.cost, model.train_op], feed_dict={model.user_input: u_input,
																						model.item_input:i_input,
																						model.prediction:ui_label.reshape((-1,1))})
					sys.stdout.write('\r {}/{} epoch, {}/{} batch, train loss:{}'.\
									format(i,e,b,total_batch,train_loss))


					if(iterations)%5000==0:
						hr_10 = val(data_loader, sess, model, tv_user, tv_item)
						print('\n',hr_10)
						if hr_10>best_hr_10:
							print( hr_10)
							best_hr_10 = hr_10
							saver.save(sess, checkpoint_dir+'model.ckpt', global_step = iterations)

		# 		hr.append(val(data_loader, sess, model, tv_user, tv_item))
		# np.save(checkpoint_dir+'hit_10_each_epoch', hr)

checkpoint_dir = './'+sys.argv[0].split('.')[0]+'_'+sys.argv[2]+'/'
if not os.path.exists(checkpoint_dir):
	os.makedirs(checkpoint_dir)

if __name__ == '__main__':
	batch_size = 256

	epochs = 100
	data_loader = Data_Loader(batch_size = batch_size)


	layers = eval('[64,16]')
	model = Model(batch_size = batch_size,
				layers = layers,
				num_user = data_loader.num_user,
				num_item = data_loader.num_item,
				hidden_size = 64,
				f_matrix = data_loader.f_matrix,
				n_neighbor_1 = data_loader.n_neighbor_1,
				n_neighbor_2 = data_loader.n_neighbor_2
				# hidden_size = 128,
				)
	if sys.argv[1]=="test":
		test(data_loader, model)
	else:
		train(batch_size, data_loader, model)
