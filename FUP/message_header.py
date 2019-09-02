from message import ISserializable
import struct # 바이너리 타입 바이트 자료들을 다루기 위해서 사용

class Header(ISserializable):
	def __init__(self,buffer):
		self.struct_fmt = '=3I2BH' # 3 unsigned int, 2byte, 1 unsigned short
		self.struct_len = struct.calcsize(self.struct_fmt)
		
		if buffer != None :
			unpacked = struct.unpack(self.struct_fmt,buffer)
			self.MSGID = unpacked[0]
			self.MSGTYPE = unpacked[1]
			self.BODYLEN = unpacked[2]
			self.FRAGMENTED = unpacked[3]
			self.LASTMSG = unpacked[4]
			self.SEQ = unpacked[5]
			
	def GetBytes(self):
		return struct.pack(
			self.struct_fmt,
			*(					 	# *() means, unpacking list/tuple
				self.MSGID,
				self.MSGTYPE,
				self.BODYLEN,
				self.FRAGMENTED,
				self.LASTMSG,
				self.SEQ
			))
	def GetSize(self):
		return self.struct_len