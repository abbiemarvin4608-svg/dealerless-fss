"""Standard standalone SIGMA configuration; explicit network s is preserved."""

SIGMA_DEFAULT_RING_BITS = 32
SIGMA_DEFAULT_FRAC_BITS = 12
SIGMA_DEFAULT_Q = 10
SIGMA_DEFAULT_TABLE_SIZE = 1 << SIGMA_DEFAULT_Q
SIGMA_BOUND = 16.0


def default_sigma_config():
    """Use Q32.12 only when the caller did not supply a network format."""
    from .fixed_point import FixedPointConfig
    return FixedPointConfig(SIGMA_DEFAULT_RING_BITS, SIGMA_DEFAULT_FRAC_BITS)


def make_sigma_activation_lut(config=None):
    """Construct the q10 table at the caller's uniform Q(ell,s) precision."""
    from .fixed_point import SigmaActivationLUT
    if config is None:
        config = default_sigma_config()
    return SigmaActivationLUT(config, q=SIGMA_DEFAULT_Q, bound=SIGMA_BOUND)
