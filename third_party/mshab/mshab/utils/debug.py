def recursive_print_dict(x):
    if isinstance(x, dict):
        for k, v in x.items():
            print(k)
            recursive_print_dict(v)
    else:
        print(type(x), x.shape)