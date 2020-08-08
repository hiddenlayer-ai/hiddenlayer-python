import os
import base64
import torch

from .client import Client

HIDDENLAYER_URL = 'http://localhost:3000/pub/v1'

# make state module / class that handles in memory stuff and loading from memory
cycleid = None
neural_net = None
optimizer = None

class Model:
	def __init__(self, model_id, secret_key):
		# try to find and load a current state (model/cycleid/etc)

		# make new client to the api
		c = Client(model_id, secret_key)

		# verify the model_id and secret_key
		resp = c.verify()

		if not resp:
			print('Not a valid model')

		self._client = c
		self._model_id = model_id
		self._secret_key = secret_key

	def forward(self, input):
		global cycleid, neural_net

		# check the most recent cycleid
		resp = self._client.get_cycleid()

		if not resp:
			print('No cycle id yet. Model has not been deployed')
			return None

		if resp != cycleid:
			# we have a new model to download
			cycleid = resp
			self._download_current_model()

		here = os.path.dirname(os.path.realpath(__file__))

		if not neural_net:
			# load the neural net and weights from disk into memory
			from .data.nn import Net
			neural_net = Net()
			neural_net.load_state_dict(torch.load(os.path.join(here, 'data', 'weights.pt')))

		return neural_net.forward(input)

	def optimize(self, input, target):
		global neural_net, optimizer

		# optimize the stored network
		pass

		# send to the blockchain if it is ready

	def _download_current_model(self):
		global neural_net, optimizer

		# download and save the most recent network
		resp = self._client.get_update()

		update = resp['update']

		here = os.path.dirname(os.path.realpath(__file__))

		if 'model' in update:
			with open(os.path.join(here, 'data', 'nn.py'), 'w') as file:
				file.write(update['model'])

			# reset in memory so it will be reloaded later
			neural_net = None

		if 'weights' in update:
			# base64 is not the most efficient. need to change this
			with open(os.path.join(here, 'data', 'weights.pt'), 'wb+') as file:
				file.write(base64.b64decode(update['weights']))

			neural_net = None

		if 'optim' in update:
			with open(os.path.join(here, 'data', 'optim.txt'), 'w') as file:
				file.write(update['optim'])

			optimizer = None
