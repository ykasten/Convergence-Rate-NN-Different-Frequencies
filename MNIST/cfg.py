print('\n*** Importing cfg.py ')

cfg = {
    # dataset
    'n_val' : 5000,
    'pi' : 50, # corruption rate

    # network
    'k' : 256,
    'depth' : 3,
    'alpha' : 0,

    # optimization
    'eta' : 0.005,
    'training_epochs' : 5001,
    'beta' : 0, # l_2 weight regularization
    'batch_size' : 100,
    'break_thresh' : 1,

    # meta
    'save_step' : 50,
    'print_step' : 100,
    'save_sbs' : False,
    'save_weights' : False,
}
