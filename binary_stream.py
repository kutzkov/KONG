from struct import *

class BinaryStream:
    """ Functions for unpacking/writing numeric values of different type from/into a binary file """
    
    def __init__(self, base_stream):
        self.base_stream = base_stream

    def read_byte(self):
        return self.base_stream.read(1)

    def read_bytes(self, length):
        return self.base_stream.read(length)

    def read_char(self):
        return self.unpack('b')

    def read_uchar(self):
        return self.unpack('B')

    def read_bool(self):
        return self.read_char() > 0
        
    def read_int16(self):
        return self.unpack('h', 2)

    def read_uint16(self):
        return self.unpack('H', 2)

    def read_int32(self):
        return self.unpack('i', 4)

    def read_uint32(self):
        return self.unpack('I', 4)

    def read_int64(self):
        return self.unpack('q', 8)

    def read_uint64(self):
        return self.unpack('Q', 8)

    def read_float(self):
        return self.unpack('f', 4)

    def read_double(self):
        return self.unpack('d', 8)

    def read_string(self):
        length = self.readUInt16()
        return self.unpack(str(length) + 's', length)

    def write_bytes(self, value):
        self.base_stream.write(value)

    def write_char(self, value):
        self.pack('c', value)

    def write_uchar(self, value):
        self.pack('C', value)

    def write_bool(self, value):
        self.pack('?', value)

    def write_int16(self, value):
        self.pack('h', value)

    def write_uint16(self, value):
        self.pack('H', value)

    def write_int32(self, value):
        self.pack('i', value)

    def write_uint32(self, value):
        self.pack('I', value)

    def write_int64(self, value):
        self.pack('q', value)

    def write_uint64(self, value):
        self.pack('Q', value)

    def write_float(self, value):
        self.pack('f', value)

    def write_double(self, value):
        self.pack('d', value)

    def write_string(self, value):
        length = len(value)
        self.writeUInt16(length)
        self.pack(str(length) + 's', value)

    def pack(self, fmt, data):
        return self.write_bytes(pack(fmt, data))

    def unpack(self, fmt, length = 1):
        return unpack(fmt, self.read_bytes(length))[0]