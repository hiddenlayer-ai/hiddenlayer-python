import os
import base64
import json
import atexit

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from .client import Client
from .encrypt import Encryptor

HERE = os.path.dirname(os.path.realpath(__file__))

N = 20
P = 0.5

class Model:
	def __init__(self, model_id, secret_key):
		# make new client to the api
		self._client = Client(model_id, secret_key)

		# verify the model_id and secret_key
		resp = self._client.verify()

		if not resp:
			print('Not a valid model')

		self._model_id = model_id
		self._secret_key = secret_key

		self._encryptor = None

		self._cycleid = None
		self._neural_net = None
		self._optimizer = None
		self._criterion = None

		# load from disk if they exist
		if os.path.exists(os.path.join(HERE, 'data', 'nn.py')):
			self._load_neural_network()

		if os.path.exists(os.path.join(HERE, 'data', 'state.json')):
			self._load_state()

		atexit.register(self._cleanup)

	def forward(self, X):
		# check the most recent cycleid
		resp = self._client.get_cycleid()

		if not resp:
			print('No cycle id yet. Model has not been deployed')
			return None

		if resp != self._cycleid:
			# we have a new model to download
			print("downloading current model")
			self._cycleid = resp
			self._download_current_model()

		if not self._neural_net:
			# load the neural net and weights from disk into memory
			self._load_neural_network()

		return self._neural_net.forward(X)

	def optimize(self, X, y):
		# optimize the stored network

		# load the optimizer into memory if it has not already been
		if not self._optimizer:
			self._load_optimizer()

		self._optimizer.zero_grad()

		y_p = self._neural_net.forward(X)

		loss = self._criterion(y_p, y)
		loss.backward()
		self._optimizer.step()

		# this is heavily in development / debugging

		# if time to encrypt and send off
		do = input("encrypt test?: ") == "yes"

		if do:
			parameters = list(self._neural_net.parameters())

			# might be able to do better list comprehesions here. check itertools.chain
			flats = np.array([layer.flatten() for layer in parameters])

			out = flats + np.random.binomial(n=N, p=P, size=len(flats)) - N * P

			'''
			out = []

			for each in flats:
				out.extend([int(10000 * val) for val in each])

			offset = -min(out)  # offset so vals to-be-encrypted will be non-negative

			# function to apply offset to each value and encrypt
			fn = lambda x: self._encryptor.encrypt(x + offset)

			# apply function
			out = list(map(fn, out + [offset]))
			'''

			print(f"[OUT]: {out}")

			# send to the blockchain

	def _cleanup(self):
		# run upon exit of program
		# store state and weights

		state = {
			'cycleid': self._cycleid,
			'public-key': self._encryptor.public_key
		}

		state_str = json.dumps(state)

		with open(os.path.join(HERE, 'data', 'state.json'), 'w') as file:
			file.write(state_str)

		torch.save(self._neural_net.state_dict(),
			os.path.join(HERE, 'data', 'weights.pt'))

	def _download_current_model(self):
		# download and save the most recent network
		resp = self._client.get_update()

		update = resp['update']

		if 'model' in update:
			with open(os.path.join(HERE, 'data', 'nn.py'), 'w') as file:
				file.write(update['model'])

			# reset in memory so it will be reloaded later
			self._neural_net = None

		if 'weights' in update:
			# base64 is not the most efficient. need to change this
			with open(os.path.join(HERE, 'data', 'weights.pt'), 'wb+') as file:
				file.write(base64.b64decode(update['weights']))

			self._neural_net = None

		if 'optim' in update:
			with open(os.path.join(HERE, 'data', 'optim.py'), 'w') as file:
				file.write(update['optim'])

			self._optimizer = None

		public_key = resp['public-key']

		self._encryptor = Encryptor(public_key)

	def _load_state(self):
		with open(os.path.join(HERE, 'data', 'state.json')) as file:
			contents = json.loads(file.read())
			self._cycleid = contents['cycleid']

			public_key = contents['public-key']
			self._encryptor = Encryptor(public_key)

	def _load_neural_network(self):
		from .data.nn import Net
		self._neural_net = Net()
		self._neural_net.load_state_dict(
			torch.load(os.path.join(HERE, 'data', 'weights.pt')))

	def _load_optimizer(self):
		# implement loading from disk
		self._optimizer = optim.SGD(self._neural_net.parameters(), lr=0.01)
		self._criterion = nn.MSELoss()