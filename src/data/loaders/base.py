from abc import ABC, abstractmethod

class AbstractDataloader(ABC):

    def __init__(self):
        pass
        
    @abstractmethod
    def train():
        raise NotImplementedError()
    
    @abstractmethod
    def val():
        raise NotImplementedError()
    
    @abstractmethod
    def test():
        raise NotImplementedError()