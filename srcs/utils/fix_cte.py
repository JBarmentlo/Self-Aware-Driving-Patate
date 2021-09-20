import logging
import time

from . import UtilsLogger


def fix_cte(simulator):
	'''
		Returns a simulator instance containing a functional env
	'''
	cte = 100
	while(abs(cte) > 1):
		state = simulator.env.reset()
		new_state, reward, done, infos = simulator.env.step([0, 1])

		if (abs(infos["cte"]) > 1):
			UtilsLogger.warn(f"Attempting to fix broken cte by driving forward a little bit. cte: {infos['cte']}")
			new_state, reward, done, infos = simulator.env.step([0, 1])
			time.sleep(0.5)
			UtilsLogger.warn(f"One step more. cte: {infos['cte']}")
		if (abs(infos["cte"]) > 1):
			new_state, reward, done, infos = simulator.env.step([0.1, 1])
			time.sleep(0.5)
			UtilsLogger.warn(f"One step more. cte: {infos['cte']}")
		if (abs(infos["cte"]) > 1):
			new_state, reward, done, infos = simulator.env.step([-0.1, 1])
			time.sleep(1)
			UtilsLogger.warn(f"One step more. cte: {infos['cte']}")
		if (abs(infos["cte"]) > 1):
			new_state, reward, done, infos = simulator.env.step([0, 1])
			time.sleep(0)
			UtilsLogger.warn(f"One step more. cte: {infos['cte']}")
		
		cte = infos["cte"]
		if (abs(cte) > 1):
			UtilsLogger.warning(f"restarting sim because cte is fucked {cte}")
			simulator.restart_simulator()

	return simulator