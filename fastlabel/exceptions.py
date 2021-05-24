class FastLabelException(Exception):
    def __init__(self, message, errcode):
        super(FastLabelException, self).__init__(
            "<Response [{}]> {}".format(errcode, message)
        )
        self.code = errcode


class FastLabelInvalidException(FastLabelException, ValueError):
    pass
