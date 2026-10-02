def Euler_E(F, U, t, dt):
	"""Integra un paso de Euler explícito para dU/dt = F(U, t)."""
	return U + dt * F(U, t)
