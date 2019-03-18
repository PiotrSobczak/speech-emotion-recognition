NUM_CLASSES = 4
EMB_DIM = 400
HIDDEN_DIM = 700
OUTPUT_DIM = 1
DROPOUT = 0.5
DROPOUT2 = 0.5
N_LAYERS = 1
BIDIRECTIONAL = False

SEQ_LEN = 25
LR = 0.001
BATCH_SIZE = 256
REG_RATIO = 0.0015
N_EPOCHS = 1000
PATIENCE = 50

VERBOSE = True


class Config:
    def __init__(self, **kwargs):
        """Network hyperparameters"""
        self.n_layers = kwargs.get("n_layers", N_LAYERS)
        self.hidden_dim = kwargs.get("hidden_dim", HIDDEN_DIM)
        self.emb_dim = kwargs.get("emb_dim", EMB_DIM)
        self.num_classes = kwargs.get("num_classes", NUM_CLASSES)
        self.dropout = kwargs.get("dropout", DROPOUT)
        self.dropout2 = kwargs.get("dropout2", DROPOUT2)
        self.bidirectional = kwargs.get("bidirectional", BIDIRECTIONAL)

        """Training hyperparameters"""
        self.seq_len = kwargs.get("seq_len", SEQ_LEN)
        self.reg_ratio = kwargs.get("reg_ratio", REG_RATIO)
        self.lr = kwargs.get("lr", LR)
        self.batch_size = kwargs.get("batch_size", BATCH_SIZE)
        self.patience = kwargs.get("patience", PATIENCE)
        self.n_epochs = kwargs.get("n_epochs", N_EPOCHS)

        """Other parameters"""
        self.verbose = kwargs.get("verbose", VERBOSE)

    def __str__(self):
        return self.to_json()

    def to_json(self):
        return self.__dict__
