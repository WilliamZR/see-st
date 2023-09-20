def add_params_hist(model, tb):
    for i, (name, param) in enumerate(model.named_parameters()):
        tb.add_histgram(name, param)