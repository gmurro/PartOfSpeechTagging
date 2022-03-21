# BiGRU instead of BiLSTM

layers_info = [
    {
        'layer_name': layers.Bidirectional,
        "layer": layers.GRU(64, return_sequences=True),
        "name": "bigru_1"
    },
    {
        "layer_name": layers.Dense,
        "units": n_classes,
        "activation": "softmax",
        "name": "output"
    }
]

# Two layers BiLSTM
layers_info = [
    {
        'layer_name': layers.Bidirectional,
        "layer": layers.LSTM(64, return_sequences=True),
        "name": "bilstm_1"
    },
    {
        'layer_name': layers.Bidirectional,
        "layer": layers.LSTM(64, return_sequences=True),
        "name": "bilstm_2"
    },
    {
        "layer_name": layers.Dense,
        "units": n_classes,
        "activation": "softmax",
        "name": "output"
    }
]

# Two Dense layers
layers_info = [
    {
        'layer_name': layers.Bidirectional,
        "layer": layers.LSTM(64, return_sequences=True),
        "name": "bilstm_1"
    },
    {
        'layer_name': layers.Dense,
        "units": num_units,
        "activation": "sigmoid",
        "name": "dense_1"
    },
    {
        "layer_name": layers.Dense,
        "units": n_classes,
        "activation": "softmax",
        "name": "output"
    }
]