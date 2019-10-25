cfg_uniform_arora = {
    # dataset
    'dim' : 2,
    'gen_x_func' : 'gen_x_arora',
    'gen_y_func' : 'gen_y_arora',
    'n_val' : 1001,
    'n_train' : 1001,
    'resample' : False,

    # network
    'n_hidden' : 5,
    'n_units' : 256,

    # optimization
    'eta' : .05,
    'n_epochs' : 100000,
    'n_batch' : 1001,
    'stop_threshold_percent' : 5,
    'max_training_time_in_minutes' : 600,
}
