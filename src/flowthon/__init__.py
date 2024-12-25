import warnings
import logging

# Suppress all UserWarning messages
warnings.filterwarnings("ignore", category=UserWarning)

# Disable all logging messages
logging.disable(logging.CRITICAL)
