import socketserver as SC
import sys

class MyTCPHandler(SC.BaseRequestHandler):
	def handle(self):
		print("클라이언트 접속 : {0}".format(self.client_address[0]))
		sock = self.request
		
		rbuff = sock.recv(1024)    # 데이터를 수신하고 bytes 형식으로 결과를 rbuff에 저장
		received = str(rbuff,encoding = "utf-8") # bytes -> string으로
		print("수신 : {0}".format(received))
		
		# 수신한 데이터를 "return"문구 붙여서 클라이언트에게 송신
		sock.send(bytes("return : " + received, encoding = "utf-8"))
		print("송신 : {0}".format("return : " + received))

if __name__ == "__main__" :
    if len(sys.argv) < 2:
        print("{0} <Bind IP>".format(sys.argv))
        bindIP = "127.0.0.1"
    else :
        bindIP = sys.argv[1]
    bindPort = 5425
    server = SC.TCPServer((bindIP,bindPort),MyTCPHandler) # IP와 포트번호를 담은 튜플, TCPHandler를 매개변수로 넘김
    print("Echo Server Start...")
    server.serve_forever()    # 클라이언트로부터 접속요청을 기다림
    
