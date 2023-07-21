import yaml

DEFAULT_FILENAME = "plasgraph_config.yaml"

PARAMS = [
    {'name': 'n_gnn_layers', 'type': int, 'default': 5},
    {'name': 'n_channels', 'type': int, 'default': 32},
    {'name': 'n_channels_preproc', 'type': int, 'default': 10},
    {'name': 'tie_gnn_layers', 'type': bool, 'default': True},
    {'name': 'preproc_activation', 'type': str, 'default': 'sigmoid'},
    {'name': 'gcn_activation', 'type': str, 'default': 'relu'},
    {'name': 'fully_connected_activation', 'type': str, 'default': 'relu'},
    {'name': 'output_activation', 'type': str, 'default': 'sigmoid'},
    {'name': 'dropout_rate', 'type': float, 'default': 0.1},
    {'name': 'loss_function', 'type': str, 'default': 'crossentropy'},
    {'name': 'l2_reg', 'type': float, 'default': 2.5e-4},
    {'name': 'features', 'type': str,
        'default': 'coverage_norm,gc_norm,kmer_dot,degree,length_norm,loglength'},
    {'name': 'n_labels', 'type': int, 'default': 2},
    {'name': 'random_seed', 'type': int, 'default': 123},
    {'name': 'plasmid_ambiguous_weight', 'type': float, 'default': 1.0},
    {'name': 'learning_rate', 'type': float, 'default': 0.005},
    {'name': 'epochs', 'type': int, 'default': 1000},
    {'name': 'early_stopping_patience', 'type': int, 'default': 100},
    {'name': 'set_thresholds', 'type': bool, 'default': False},
    {'name': 'plasmid_threshold', 'type': float, 'default': 0.5},
    {'name': 'chromosome_threshold', 'type': float, 'default': 0.5},
    {'name': 'minimum_contig_length', 'type': int, 'default': 100}
]


class config:
    def __init__(
        self,
        yaml_file=None
    ):
        # create list of parameter values and assign default value for each
        self._params = {}
        for param_dict in PARAMS:
            self._params[param_dict['name']] = param_dict['default']

        # read config file in YAML format and rewrite default values
        if yaml_file is not None:
            # read the whole file
            with open(yaml_file) as file:
                yaml_parameters = yaml.safe_load(file)
            # check each known parameter if it was in the file
            for param_dict in PARAMS:
                name = param_dict['name']
                if name in yaml_parameters:
                    # convert value to the correct type
                    value = yaml_parameters[name]
                    cast_value = param_dict['type'](value)
                    self._params[name] = cast_value
            known_params = {param_dict['name'] for param_dict in PARAMS}
            for name in yaml_parameters:
                if name not in known_params:
                    raise ValueError(f"Parameter {name} from {yaml_file} is not known")
        # special handling of features param (split to a list)
        self._params["features"] = tuple(self._params["features"].split(','))
        # special handling of output_activation - change to None if indicated
        if self._params["output_activation"].casefold() == "none".casefold():
            self._params["output_activation"] = None
    
    def __getitem__(self, key):
        return self._params[key]

    def __setitem__(self, key, value):
        assert key in self._params
        self._params[key] = value

    def write_yaml(self, filename):
        to_write = self._params.copy()
        to_write["features"] = ",".join(to_write["features"])
        with open(filename, "w") as file:
            yaml.dump(to_write, file)

    
