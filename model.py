import abc


# noinspection PyClassHasNoInit
class Model:
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def params(self):
        pass

    @abc.abstractmethod
    def forward(self, x):
        pass

    @abc.abstractmethod
    def create_in_layer(self, in_size):
        pass

    @abc.abstractmethod
    def create_out_layer(self, in_size):
        pass
