def encode_commitment(text, kind):
    return {'text': text, 'kind': kind}


def make_constraint(items):
    return {'constraint': list(items)}
