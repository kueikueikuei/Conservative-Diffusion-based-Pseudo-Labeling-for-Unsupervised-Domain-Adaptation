from .digit_LeNet import DigitLeNet
from .digit_Svhn2Mnist import DigitSvhn2Mnist
from .office_ResNet50 import Office31ResNet
from .imageCLEF_AlexNet import ImageCLEFAlexNet
from .visda_ResNet50 import VISDA17ResNet
from .cifar10_13layer import Cifar10_13layer
from .ResNet50 import ResNet50
def build_network(net_name):
    """Builds the neural network."""

    implemented_networks = ('digit_LeNet', 'office_ResNet50','digit_Svhn2Mnist','ImageCLEF_AlexNet','visda_ResNet50','cifar10_13layer')
    assert net_name in implemented_networks

    net = None

    if net_name == 'digit_LeNet':
        net = DigitLeNet()
        net.rep_dim=100
    if net_name == 'office_ResNet50':
        net = Office31ResNet()
        net.rep_dim=2048
    if net_name == 'visda_ResNet50':
        net = VISDA17ResNet()
        net.rep_dim=2048
    if net_name == 'ImageCLEF_AlexNet':
        net = ImageCLEFAlexNet()
        net.rep_dim=4096
    if net_name == 'digit_Svhn2Mnist':
        net = DigitSvhn2Mnist()
        net.rep_dim=2048
    if net_name == 'cifar10_13layer':
        net = Cifar10_13layer()
        net.rep_dim=128

    return net