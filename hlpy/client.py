import requests
import json

HIDDENLAYER_URL = 'http://localhost:3000/pub/v1'

class Client:
	def __init__(self, model_id, secret_key):
		self._model_id = model_id
		self._secret_key = secret_key

	def verify(self):
		r = self._common_api('verify')
		return json.loads(r.content)

	def get_cycleid(self):
		r = self._common_api('cycleid')
		return json.loads(r.content)

	def get_update(self):
		r = self._common_api('update')
		return json.loads(r.content)

	def _common_api(self, resource):
		url = f'{HIDDENLAYER_URL}/{self._model_id}/{resource}'
		headers = {'x-secret-key': self._secret_key}

		return requests.get(url, headers=headers)