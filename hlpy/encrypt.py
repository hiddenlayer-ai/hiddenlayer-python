import random
import math

import gmpy2

class Encryptor:
	def __init__(self, public_key):
		self._public_key = public_key
		self._n = int(self._public_key, 16)
		self._g = self._n + 1
		self._n2 = self._n * self._n

	def encrypt(self, plain_text):
		m = plain_text

		# check for errors and such

		r = Encryptor._get_random(self._n)
		r = int(gmpy2.powmod(r, self._n, self._n2))

		m = int(gmpy2.powmod(self._g, m, self._n2))

		c = m * r
		return c % self._n2

	@property
	def public_key(self):
		return self._public_key
	
	@staticmethod
	def _get_random(n):
		gcd = 0
		r = 0

		while gcd != 1:
			r = random.SystemRandom().randrange(1, n)

			gcd = math.gcd(r, n)

		return r