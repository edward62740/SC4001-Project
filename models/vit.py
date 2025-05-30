import ml_collections


def get_testing():
	"""Returns a minimal configuration for testing."""
	config = ml_collections.ConfigDict()
	config.patches = ml_collections.ConfigDict({'size': (16, 16)})
	config.hidden_size = 1
	config.transformer = ml_collections.ConfigDict()
	config.transformer.mlp_dim = 1
	config.transformer.num_heads = 1
	config.transformer.num_layers = 1
	config.transformer.attention_dropout_rate = 0.0
	config.transformer.dropout_rate = 0.1
	config.classifier = 'token'
	config.representation_size = None
	return config


def get_b16_config():
	"""Returns the ViT-B/16 configuration."""
	config = ml_collections.ConfigDict()
	config.patches = (16, 16)
	config.hidden_size = 768
	config.mlp_dim = 3072
	config.num_heads = 12
	config.num_layers = 12
	config.att_dropout = 0.0
	config.dropout_rate = 0.1
	config.classifier = 'token'
	return config


# def get_b16_config():
#     """Returns the ViT-B/16 configuration."""
#     config = ml_collections.ConfigDict()
#     config.patches = (16,16)
#     config.hidden_size = 768
#     config.mlp_dim = 3072
#     config.num_heads = 12
#     config.num_layers = 12
#     config.att_dropout = 0.0
#     config.dropout = 0.1
#     config.classifier = 'token'
#     return config


def get_b32_config():
	"""Returns the ViT-B/32 configuration."""
	config = get_b16_config()
	config.patches = (16, 16)
	return config


def get_l16_config():
	"""Returns the ViT-L/16 configuration."""
	config = ml_collections.ConfigDict()
	config.patches = ml_collections.ConfigDict({'size': (16, 16)})
	config.hidden_size = 1024
	config.transformer = ml_collections.ConfigDict()
	config.transformer.mlp_dim = 4096
	config.transformer.num_heads = 16
	config.transformer.num_layers = 24
	config.transformer.attention_dropout_rate = 0.0
	config.transformer.dropout_rate = 0.1
	config.classifier = 'token'
	config.representation_size = None
	return config


def get_l32_config():
	"""Returns the ViT-L/32 configuration."""
	config = get_l16_config()
	config.patches.size = (32, 32)
	return config


def get_h14_config():
	"""Returns the ViT-L/16 configuration."""
	config = ml_collections.ConfigDict()
	config.patches = ml_collections.ConfigDict({'size': (14, 14)})
	config.hidden_size = 1280
	config.transformer = ml_collections.ConfigDict()
	config.transformer.mlp_dim = 5120
	config.transformer.num_heads = 16
	config.transformer.num_layers = 32
	config.transformer.attention_dropout_rate = 0.0
	config.transformer.dropout_rate = 0.1
	config.classifier = 'token'
	config.representation_size = None
	return config


def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = (16,16)
    config.hidden_size = 768
    config.mlp_dim = 3072
    config.num_heads = 12
    config.num_layers = 12
    config.att_dropout = 0.0
    config.dropout_rate = 0.1
    config.classifier = 'token'
    return config



def get_dinob16_config():
    """Returns the DINOv2-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = (14,14)
    config.hidden_size = 768
    config.mlp_dim = 3072
    config.num_heads = 12
    config.num_layers = 12
    config.att_dropout = 0.0
    config.dropout_rate = 0.0
    config.classifier = 'token'
    config.ls = True
    return config

def get_dinos16_config():
	"""Returns the DINOv2-S/16 configuration."""
	config = ml_collections.ConfigDict()
	config.patches = (14,14)
	config.hidden_size = 384
	config.mlp_dim = 1536
	config.num_heads = 6
	config.num_layers = 12
	config.att_dropout = 0.0
	config.dropout_rate = 0.0
	config.classifier = 'token'
	config.ls = True
	return config