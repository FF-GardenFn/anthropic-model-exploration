def plan_cache():
    cache = {}
    def get(sig):
        return cache.get(sig)
    def put(sig, plan):
        cache[sig] = plan
    return get, put


def route(query, circuits):
    get, put = plan_cache()
    sig = str(hash(query) % 100000)
    plan = get(sig)
    if plan is None and circuits:
        plan = {'circuits': [circuits[0]], 'cost': circuits[0].get('cost', 0.0)}
        put(sig, plan)
    return plan