"""This module contains useful tools for image manipulation.

It's used to read the header. And save the image in a nice format.

Deprecated 12.06.2020
"""

import os.path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from bitarray import bitarray
from bitstring import BitArray



def header_reader(file_path):
    if os.path.isfile(file_path):
        with open(file_path, "rb") as f:
            b = bitarray()
            b.fromfile(f)
            header = b.tobytes()[:b.tobytes().find(b'\x03')].decode()
            return header
    else:
        raise FileNotFoundError


def body_reader(file_path):
    if os.path.isfile(file_path):
        with open(file_path, "rb") as f:
            b = bitarray()
            b.fromfile(f)
            body = b.tobytes()[b.tobytes().find(b'\x03')+1:]
            return body
    else:
        raise FileNotFoundError


def interpreter(bitarray):
    if len(bitarray) != 16:
        raise ValueError
    else:
        bitarray_body = bitarray[12:]
        bitarray_body.append(bitarray[:8])
        bitarray_identifier = bitarray[8:12]

    if bitarray_body.uint == 2500 and bitarray_identifier.uint == 2:
        return -1
    else:
        return bitarray_body.uint


class BinHeader:
    """Header Object for a file.

    """
    def __init__(self, file_path):
        self._raw_header = header_reader(file_path=file_path)


class BinBody:
    """Body Object for a file.

    """
    def __init__(self, file_path):
        self._raw_body = body_reader(file_path=file_path)
        self._bitarray = BitArray(self._raw_body)
        self._np_array = None

    @property
    def np_array(self):
        if self._np_array is None:
            print('initializing array')
            self._np_array = np.empty(810000, dtype=np.int16)
            print('array initialized')
            for cell in range(900*900):
                print('processing cell {} of 810000'.format(cell))
                value = interpreter(self._bitarray[(cell*2)*8:((cell+1)*2)*8])
                self._np_array[cell] = value

            self._np_array = self._np_array.reshape(900, 900)

        return self._np_array

    @property
    def image(self):
        plt.imshow(self.np_array, cmap=plt.cm.coolwarm, origin='lower')
        plt.show()
        return True