from .custom_logger import CustomLogger

# Create a global logger instance
GLOBAL_LOGGER = CustomLogger().get_logger("ecomm-prod-assistant")
