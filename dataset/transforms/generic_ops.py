class KeepKeys(object):
    def __init__(self, keep_keys, **kwargs):
        self.keep_keys = keep_keys

    def __call__(self, data):
        keys = list(data.keys())
        for k in keys:
            if k not in self.keep_keys:
                data.pop(k)
        return data