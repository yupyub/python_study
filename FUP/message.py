# FUP : File Up Protocol

# Message type 상수 정의
REQ_FILE_SEND = 0x01 # Request
REP_FILE_SEND = 0x02 # Response 
FILE_SEND_DATA = 0x03 
FILE_SEND_RES = 0x04 

# File fragmented 여부 상수
NOT_FRAGMENTED = 0x00 
FRAGMENTED = 0x01 

# 분할된 메시지의 마지막 메시지 여부 <LASTMSG>
NOT_LASTMSG = 0x00
LASTMSG = 0x01

# 파일전송 수락 여부 <RESPONSE> 상수 정의
ACCEPTED = 0x00
DENIED = 0x01

# 파일 전송 성공 여부 <RESULT> 상수 정의
FAIL = 0x00
SUCCESS = 0x01


# 메시지,헤더,바디 가 공통으로 상속 <자바의 interface 격>
# 이들은 반드시 자신의 데이터를 바이트 배열로 반환하고, 
# 그 바이트 배열의 크기를 반환 해야 한다
class ISerializable:
	def GetBytes(self):
		pass
	def GetSize():
		pass
	
class Message(ISerializable):
	def __init__(self):
		self.Header = ISerializable()
		self.Body = ISerializable()
		
	def GetBytes(self):
		buffer = bytes(self.GetSize())
		header = self.Header.GetBytes()
		body = self.Body.GetBytes()
		return header + body
	
	def GetSize(self):
		return self.Header.GetSize() + self.Body.GetSize()

	