class FastLabelException(Exception):
    def __init__(self, message, errcode=None):
        super(FastLabelException, self).__init__(
            "<Response [{}]> {}".format(errcode, message)
        )
        self.message = message
        self.code = errcode

    def __reduce__(self):
        return (self.__class__, (self.message, self.code))


class FastLabelInvalidException(FastLabelException, ValueError):
    pass
