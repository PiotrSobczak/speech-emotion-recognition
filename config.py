NUM_CLASSES = 4


class Config:
    def __str__(self):
        return self.to_json()

    def to_json(self):
        return self.__dict__


class LinguisticConfig(Config):
    def __init__(self, **kwargs):
        """Network hyperparameters"""
        self.n_layers = kwargs.get("n_layers", 1)
        self.hidden_dim = kwargs.get("hidden_dim", 200)
        self.emb_dim = kwargs.get("emb_dim", 400)
        self.num_classes = kwargs.get("num_classes", NUM_CLASSES)
        self.dropout = kwargs.get("dropout", 0.75)
        self.dropout2 = kwargs.get("dropout2", 0.4)
        self.bidirectional = kwargs.get("bidirectional", False)

        """Training hyperparameters"""
        self.seq_len = kwargs.get("seq_len", 25)
        self.reg_ratio = kwargs.get("reg_ratio", 0.0015)
        self.lr = kwargs.get("lr", 0.001)
        self.batch_size = kwargs.get("batch_size", 128)
        self.patience = kwargs.get("patience", 30)
        self.n_epochs = kwargs.get("n_epochs", 1000)

        """Other parameters"""
        self.verbose = kwargs.get("verbose", False)
        self.model_weights_name = "linguistic_model.torch"
        self.model_config_name = "linguistic_model.json"

    @staticmethod
    def from_json(config_json):
        cfg = LinguisticConfig()
        cfg.hidden_dim = config_json["hidden_dim"]
        return cfg


class AcousticLLDConfig(Config):
    def __init__(self, **kwargs):
        """Network hyperparameters"""
        self.n_layers = kwargs.get("n_layers", 1)
        self.hidden_dim = kwargs.get("hidden_dim", 100)
        self.emb_dim = kwargs.get("emb_dim", 34)
        self.num_classes = kwargs.get("num_classes", NUM_CLASSES)
        self.dropout = kwargs.get("dropout", 0.0)
        self.dropout2 = kwargs.get("dropout2", 0.0)
        self.bidirectional = kwargs.get("bidirectional", False)

        """Training hyperparameters"""
        self.seq_len = kwargs.get("seq_len", 200)
        self.reg_ratio = kwargs.get("reg_ratio", 0.0015)
        self.lr = kwargs.get("lr", 0.001)
        self.batch_size = kwargs.get("batch_size", 256)
        self.patience = kwargs.get("patience", 30)
        self.n_epochs = kwargs.get("n_epochs", 1000)

        """Other parameters"""
        self.verbose = kwargs.get("verbose", False)
        self.model_weights_name = "acoustic_lld_model.torch"
        self.model_config_name = "acoustic_ld_model.json"

    @staticmethod
    def from_json(config_json):
        cfg = AcousticLLDConfig()
        cfg.hidden_dim = config_json["hidden_dim"]
        return cfg


class AcousticSpectrogramConfig(Config):
    def __init__(self, **kwargs):
        """Network hyperparameters"""
        self.fc_size = kwargs.get("hidden_dim", 50)
        self.conv_size = kwargs.get("conv_size", 3)
        self.pool_size = kwargs.get("pool_size", 2)
        self.num_filters = kwargs.get("num_filters", [8, 8, 8, 8])
        self.num_classes = kwargs.get("num_classes", NUM_CLASSES)
        self.dropout = kwargs.get("dropout", 0.5)

        """Training hyperparameters"""
        self.reg_ratio = kwargs.get("reg_ratio", 0.0)
        self.lr = kwargs.get("lr", 0.001)
        self.batch_size = kwargs.get("batch_size", 128)
        self.patience = kwargs.get("patience", 30)
        self.n_epochs = kwargs.get("n_epochs", 1000)

        """Other parameters"""
        self.verbose = kwargs.get("verbose", False)
        self.model_weights_name = "acoustic_spec_model.torch"
        self.model_config_name = "acoustic_spec_model.json"

    @staticmethod
    def from_json(config_json):
        cfg = AcousticSpectrogramConfig()
        return cfg


class EnsembleConfig(Config):
    def __init__(self, acoustic_config, linguistic_config):
        self.acoustic_config = acoustic_config
        self.linguistic_config = linguistic_config
        self.dropout = 0.7
        self.model_weights_name = "ensemble_model.torch"
        self.model_config_name = "ensemble_model.json"
        self.patience = 10


    @staticmethod
    def from_json(config_json):
        cfg = AcousticSpectrogramConfig()
        cfg.acoustic_config = AcousticSpectrogramConfig.from_json(config_json["acoustic_config"])
        cfg.linguistic_config = LinguisticConfig.from_json(config_json["linguistic_config"])
        cfg.dropout = config_json["dropout"]
        return cfg

    def to_json(self):
        json_obj = {}
        json_obj["dropout"] = self.dropout
        json_obj["acoustic_config"] = self.acoustic_config.to_json()
        json_obj["linguistic_config"] = self.linguistic_config.to_json()
        return json_obj
