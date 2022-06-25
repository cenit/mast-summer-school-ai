#!/usr/bin/env python3

from random import seed
from random import random

num_facce=4
estrazioni = []
i = 0
while True:
  random_number = int(random()*num_facce+1)
  set_estrazioni = set(estrazioni)
  if random_number in set_estrazioni:
    continue
  else:
    estrazioni.append(random_number)
  if len(estrazioni) == 4:
    break

print(f'Ordine estrazione: {estrazioni}')
