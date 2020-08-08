import os
import base64
import torch

from .client import Client

HERE = os.path.dirname(os.path.realpath(__file__))

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

		self._cycleid = None
		self._neural_net = None
		self._optimizer = None

	def forward(self, input):
		# check the most recent cycleid
		resp = self._client.get_cycleid()

		if not resp:
			print('No cycle id yet. Model has not been deployed')
			return None

		if resp != self._cycleid:
			# we have a new model to download
			self._cycleid = resp
			self._download_current_model()

		if not self._neural_net:
			# load the neural net and weights from disk into memory
			self._load_neural_network()

		return self._neural_net.forward(input)

	def optimize(self, input, target):
		# optimize the stored network
		# send to the blockchain if it is ready
		pass

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

	def _load_state(self):
		with open(os.path.join(HERE, 'data', 'state.json')) as file:
			contents = json.loads(file.read())
			self._cycleid = contents['cycleid']

	def _load_neural_network(self):
		from .data.nn import Net
		self._neural_net = Net()
		self._neural_net.load_state_dict(
			torch.load(os.path.join(HERE, 'data', 'weights.pt')))

	def _load_optimizer(self):
		from .data.optim import optim
		self._optimizer = optim