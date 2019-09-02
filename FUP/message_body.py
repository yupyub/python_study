from message import ISerializeble
import message
import struct
class BodyRequest(ISerializeble) :
	def __init__(self,buffer):
		if buffer != None :
			slen = len(buffer)
			
			# 1 unsigned long long, N character
			self.struct_fmt = str.format("=Q{0}s",slen-8)
			self.struct_len = struct.calcsize(self.struct_fmt)
			if slen > 4: # unsigned long long 의 크기
				slen = slen - 4
			else slen = 0
			
			unpacked = struct.unpack(slef.struct_fmt,buffer)
			
			self.FILESIZE = unpacked[0]
			self.FILENAME = unpacked[1].decode(encoding = "utf-8").replace("\x00","")
		else:
			self.struct_fmt = str.format("=Q{0}s",0)
			self.struct_len = struct.calcsize(self.struct_fmt)
			self.FILE_SIZE = 0 
			self.FILENAME = ""	
			
	def GetSize(self):
		
	